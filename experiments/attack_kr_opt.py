import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from pathlib import Path
from memit.memit_main import upd_matrix_match_shape
import os
import torch.nn.functional as F

from util.globals import STATS_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 模拟 nethook
class nethook:
    @staticmethod
    def get_parameter(model, w_name):
        module_path, param_name = w_name.rsplit('.', 1)
        mod = model.get_submodule(module_path)
        return getattr(mod, param_name)

    class Trace:
        def __init__(self, module, layer_name, edit_input=None, retain_output=False, stop=False):
            self.module = module
            self.layer_name = layer_name
            self.edit_input_fn = edit_input
            self.hook = None

        def __enter__(self):
            if self.edit_input_fn:
                def pre_hook_fn(module, inp):
                    return self.edit_input_fn(inp[0], self.layer_name)

                self.hook = self.module.get_submodule(self.layer_name).register_forward_pre_hook(pre_hook_fn)
            else:
                raise NotImplementedError("nethook.Trace only satisfy edit_input")

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.hook:
                self.hook.remove()



# --- 第 2 部分：【已修正】的 MEMIT 求解器 ---

def simulate_memit_solve_CORRECT(K, R_resid, C, hparams):
    """
    【已修正】
    可微的 MEMIT 求解器，逻辑与您的 execute_memit 一致。
    """
    w = hparams["mom2_update_weight"]

    # 1. 求解 adj_k = (wC + KK^T)^-1 @ K
    adj_k = torch.linalg.solve(
        w * C + K @ K.T,
        K
    )  # (D_int, 1)

    # 2. 计算 upd_matrix = R_resid @ adj_k.T
    upd_matrix_pre_transpose = R_resid @ adj_k.T  # (D_hid, 1) @ (1, D_int) -> (D_hid, D_int)

    return upd_matrix_pre_transpose


def get_logits_from_k(model, tok, k_vector, config):
    """
    【新】: 将 k_vector 注入到指定层，并运行模型的剩余部分
    """

    # dummy_input_ids = tok(tok.bos_token, return_tensors="pt").to("cuda")
    
    dummy_input_ids = tok("The official religion of Edwin of Northumbria is", return_tensors="pt").to("cuda")

    k_vector_reshaped = k_vector.squeeze().view(1, 1, -1)  # 形状 (1, 1, D_int)

    # 3. 定义“钩子”函数 (Hook Function)
    target_layer_name = config.rewrite_module_tmp.format(config.final_layer_to_attack)

    def edit_input_hook(inp, layer_name):
        return (k_vector_reshaped,)

    with torch.no_grad():
        with nethook.Trace(
            model,
            target_layer_name,  # 钩住 down_proj 层
            edit_input=edit_input_hook,  # 使用我们的替换函数
        ) as tr:
            outputs = model(**dummy_input_ids)
        #outputs =model(**dummy_input_ids)

    # 5. 提取最终的 Logits
    logits = outputs.logits[:, -1, :]  # 形状 (1, V)
    return logits.squeeze()  # 形状 (V,)


def print_top_10(logits, tok):
    """
    【新】: 计算 softmax, 获取 top 10, 并打印
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 100)

    top_tokens = tok.convert_ids_to_tokens(top_indices.cpu())

    print("Rank | Token          | Probability")
    print("-----|----------------|--------------")
    for i, (prob, token) in enumerate(zip(top_probs, top_tokens)):
        print(f" {i + 1: <4}| {token: <14} | {prob.item():.4f}")


# --- 第 3 部分：【已修正】的“精简版”攻击代码 ---
COV_CACHE = {}
from rome.layer_stats import layer_stats
def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    # 修复：缓存键应该包含数据集和样本大小，以区分不同的协方差矩阵设置
    key = (model_name, layer_name, mom2_dataset, mom2_n_samples)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name} (dataset={mom2_dataset}, n_samples={mom2_n_samples}).")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            # "data/stats_estimated",
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            # sample_size=mom2_n_samples,
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )

def run_attack_simple_k_r_CORRECT(config: SimpleNamespace,model,tok):

    print(f"target model: {config.model_name} ---")

    # 1. 加载模型和分词器
    print("model and tokenizer init...")
    # model = AutoModelForCausalLM.from_pretrained(config.model_name).cuda()
    # tok = AutoTokenizer.from_pretrained(config.model_name)
    target_layer_name = config.rewrite_module_tmp.format(config.final_layer_to_attack)

    # 2. 加载观测到的 Delta
    print(f"import delta: {config.delta_file_path}")
    delta_observed_dict = torch.load(config.delta_file_path)
    target_weight_name = target_layer_name + ".weight"

    target_shape = nethook.get_parameter(model, target_weight_name).shape

    delta_observed_8 = delta_observed_dict[target_weight_name].cuda().double()

    # D_int, D_hid = delta_observed_8.shape

    if delta_observed_8.shape != target_shape:
        if delta_observed_8.T.shape == target_shape:
            print("wran: Delta shape is transpose, auto-transposing...")
            delta_observed_8 = delta_observed_8.T
        else:
            raise ValueError(f"Delta shape {delta_observed_8.shape} is not match with {target_shape}")

    D_hid_arch, D_int_arch = target_shape
    print(f"D_hid: {D_hid_arch}, D_int: {D_int_arch} (match)")

    # For gpt2-xl, transpose delta before SVD to get correct key matrix
    # gpt2-xl stores weights as (din, dout), but SVD needs (dout, din) to extract key vectors
    needs_transpose = getattr(config, 'needs_transpose', False)
    if needs_transpose:
        print(f"Transposing delta matrix for gpt2-xl before SVD...")
        delta_observed_8 = delta_observed_8.T

    # --- 注释掉 Ground Truth 读取逻辑 ---
    # print(f"load K/R (Ground Truth): {config.kr_ground_truth_path}")
    # kr_true_dict = torch.load(config.kr_ground_truth_path)
    # k_true_key = f"{target_layer_name}.k_true"
    # r_true_key = f"{target_layer_name}.r_true"
    # k_pre_true_key = f"{target_layer_name}.k_pre_true"

    # if k_true_key not in kr_true_dict or r_true_key not in kr_true_dict:
    #     raise ValueError(f"error:  {config.kr_ground_truth_path} lack '{k_true_key}' or '{r_true_key}'")

    # k_true = kr_true_dict[k_true_key].cuda().double()
    # k_true_norm = torch.norm(k_true)
    # r_true = kr_true_dict[r_true_key].cuda().double()  # 形状 (D_hid, 1)
    # r_true_norm = torch.norm(r_true)  # 标量
    # k_pre_true = kr_true_dict[k_pre_true_key].cuda().double()
    # k_pre_true_norm = torch.norm(k_pre_true)
    # print("success load K and R for evaluate。")

    # 3. (SVD 降维) 执行 SVD
    print(f"for {target_weight_name} execute SVD...")
    U, S, Vh = torch.linalg.svd(delta_observed_8, full_matrices=False)
    v1_r_direction = U[:, 0].double().cuda()
    v1_k_post_direction = Vh[0, :].double().cuda().unsqueeze(1)

    # --- 注释掉评估模块 1（依赖 Ground Truth）---
    # print("\n--- [evaluate module 1: SVD R direction] ---")
    # # (确保 v1_r_direction 和 r_true 都是 (D_hid,) 形状)
    # r_true_direction = (r_true.squeeze() / (r_true_norm + 1e-9)).float()
    # k_true_direction = (k_true.squeeze()/(k_true_norm+1e-9)).float()
    # # 计算余弦相似度
    # # r_cos_sim = F.cosine_similarity(v1_r_direction, r_true_direction, dim=0)

    # # print(f"SVD's R vs true R direction: {r_cos_sim.item():.4f}")
    # # if r_cos_sim.item() < 0.9 and r_cos_sim.item()>-0.9:
    # #     print("error SVD R differs true R")
    # # else:
    # #     # if r_cos_sim.item()<0:
    # #         # v1_r_direction = -v1_r_direction
    # #         # v1_k_post_direction = -v1_k_post_direction
    # #         # r_cos_sim = F.cosine_similarity(v1_r_direction, r_true_direction, dim=0)
    # #         # print(f"SVD's R vs true R direction: {r_cos_sim.item():.4f}")
    # #         # k_cos_sim = F.cosine_similarity(v1_k_post_direction.squeeze(),k_true_direction,dim=0)
    # #         # print(f"SVD's K vs true K direction: {k_cos_sim.item():.4f}")
    # #     print("good R direction")
    # # print("--------------------------------------\n")

    # 4. 加载协方差矩阵 C (!!! 使用真实的 get_cov !!!)
    print(f"8th layer C matrix...")
    C_matrix_8 = get_cov(model, tok, target_layer_name,
                         mom2_dataset=config.mom2_dataset,
                         mom2_n_samples=config.mom2_n_samples,
                         mom2_dtype=config.mom2_dtype)
    
    C_matrix_8 = C_matrix_8.double()

    w = config.mom2_update_weight

    # --- [New Logic: Rank N Support] ---
    target_rank = getattr(config, 'target_rank', 1)
    
    if target_rank > 1:
        print(f"\n--- [Rank {target_rank} Mode Detected: Subspace Recovery] ---")
        # 1. 获取前 N 个右奇异向量 (Vh 的行是奇异向量)
        # Vh shape: (D_int, D_int) (full_matrices=False, usually (K, N))
        print("Using top-N singular vectors...")
        V_sub = Vh[:target_rank, :].double().cuda() # (Rank, D_int)

        # 2. 反解子空间 K_pre_subspace = (w * C) @ V^T
        # Result shape: (D_int, Rank)
        print("Recovering subspace with C matrix...")
        k_pre_guess_subspace = (w * C_matrix_8) @ V_sub.T 

        # 3. QR 分解获取正交基
        print("Applying QR decomposition for orthogonal basis...")
        Q, R_qr = torch.linalg.qr(k_pre_guess_subspace, mode='reduced')
        
        # Q 是正交基 (D_int, Rank)
        k_attack_result = Q.detach()

        print("Subspace recovery complete. Skipping optimization loop.")
        print("------------------------------------------------------\n")
        
        # 只返回 k_attack_result
        return k_attack_result

    # --- [Rank 1 Logic (Original)] ---
    print("Solving for k_pre direction using SVD (k_post) and C matrix...")
    with torch.no_grad():
        k_pre_direction_guess = (w * C_matrix_8) @ v1_k_post_direction
        k_pre_direction_guess = k_pre_direction_guess / (torch.norm(k_pre_direction_guess) + 1e-9)

    print("init k_attack_8 and r (using SVD + C)")

    r = torch.tensor([abs(S[0].item())], device="cuda", requires_grad=True)
    k_attack_8 = k_pre_direction_guess.clone().detach().requires_grad_(True)
    #k_attack_8 = torch.randn_like(k_pre_direction_guess).requires_grad_(True)

    # 6. 设置优化器
    optimizer = torch.optim.Adam([k_attack_8, r], lr=config.learning_rate_r)
    hparams_dict_solve = {"mom2_update_weight": config.mom2_update_weight}

    print("--- [optimize k and r)] ---")

    for it in range(config.iterations):
        optimizer.zero_grad()

        # (A) 模拟 R_resid_8
        R_resid_8 = (r * v1_r_direction.unsqueeze(1)).double()  # (D_hid, 1)

        # (B) 模拟 Delta_8 (使用修正后的求解器)
        delta_attack_8_pre_transpose = simulate_memit_solve_CORRECT(
            k_attack_8,  # (D_int, 1)
            R_resid_8,  # (D_hid, 1)
            C_matrix_8,  # (D_int, D_int)
            hparams_dict_solve
        )  # (D_hid, D_int)

        # (C) 匹配形状 (使用真实的 upd_matrix_match_shape)
        delta_attack_8 = upd_matrix_match_shape(
            delta_attack_8_pre_transpose,
            delta_observed_8.shape  # 目标形状 (D_int, D_hid)
        )

        # (D) 计算总损失
        L_recon = torch.norm(delta_observed_8 - delta_attack_8) ** 2
        L_knowledge = torch.norm(delta_observed_8 @ k_attack_8 - R_resid_8) ** 2
        L_k_l2 = config.lambda_l2 * torch.norm(k_attack_8)**2
        L_total = L_recon + config.beta * L_knowledge + L_k_l2

        # (E) 反向传播和优化
        L_total.backward()
        optimizer.step()

        if it % 50 == 0:
            print(f"Iter {it}: Loss={L_total.item():.6f} ...")
            print(f"  r = {r.item():.4f}")
            # --- 注释掉依赖 Ground Truth 的评估 ---
            # k_pre_true_cos_sim = F.cosine_similarity(k_attack_8.squeeze(), k_pre_true.squeeze(), dim=0)
            # print(f"  k_cos_sim = {k_pre_true_cos_sim.item():.6f}")
            # with torch.no_grad():
            #     k_mse = F.mse_loss(k_attack_8, k_pre_true)
            # print(f"  k_mse = {k_mse.item():.6f}")
            # print(f"  k_attack_norm = {torch.norm(k_attack_8).item():.6f} (true k_pre_norm: {k_pre_true_norm.item():.6f})")
            print(f"  k_attack_norm = {torch.norm(k_attack_8).item():.6f}")

    print("--- [attack finish] ---")
    # 只返回 k_attack_8
    return k_attack_8.detach()


if __name__ == "__main__":

    # 1. 【您必须修改这里】: 定义您的攻击配置
    # ----------------------------------------------------
    case_id = "1"
    config = SimpleNamespace(
        # -- 模型和文件路径 --
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        delta_file_path=f"orth_defence_edit_memit_amount/edit_amounts_batch_case_{case_id}.pt",
        kr_ground_truth_path=f"orth_defence_edit_memit_amount/kr_ground_truth_case_{case_id}.pt",

        # -- 层级模板 --
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        final_layer_to_attack=4,  # (占位符) 替换为您的目标层

        # -- C 矩阵 (get_cov) 参数 --
        # (STATS_DIR 在脚本顶部设置)
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",

        # -- 求解器 (MEMIT) 参数 --
        mom2_update_weight=15000,

        # -- 优化器参数 --
        learning_rate_r=1e-2,  # (占位符) K 和 r 的学习率
        iterations=0,  # (占位符) 迭代次数
        beta=5e-6,
        lambda_l2=5e-6
    )
    # ----------------------------------------------------

    # 2. 检查 CUDA
    if not torch.cuda.is_available():
        print("error：CUDA not available")
    elif not Path(config.delta_file_path).exists():
        print(f"error: cant find Delta : {config.delta_file_path}")
    elif not Path(config.kr_ground_truth_path).exists():
        print(f"error: cant find K/R : {config.kr_ground_truth_path}")
    else:
        print("CUDA available...")

        print("Loading model and tokenizer (once)...")
        model = AutoModelForCausalLM.from_pretrained(config.model_name).cuda()
        tok = AutoTokenizer.from_pretrained(config.model_name)
        print("Model and tokenizer loaded.")

        # 3. 【启动】: 调用攻击函数
        try:
            # 只接收 k_attack_result
            optimized_k = run_attack_simple_k_r_CORRECT(config, model, tok)

            # 4. 评估模块（不再依赖 Ground Truth）
            print("\n--- [final result] ---")

            # (a) 评估 K
            k_cos_sim = F.cosine_similarity(optimized_k.squeeze(), k_pre_true.squeeze(), dim=0)
            print(f"optimized K vs true K (cos sim): {k_cos_sim.item():.6f}")
            optimized_k_norm = torch.norm(optimized_k).item()
            print(f"optimized K norm: {optimized_k_norm:.6f}")
            print(f"true K_pre norm: {torch.norm(k_pre_true).item():.6f}")
            # (b) 评估 R (幅度)
            print(f"optimized R (r): {optimized_r:.6f}")
            print(f"true R norm: {r_true_norm.item():.6f}")
            r_diff_percent = (optimized_r - r_true_norm.item()) / (r_true_norm.item() + 1e-9)
            print(f"norm diff: {r_diff_percent * 100:.6f}%")
            print("----------------------------------\n")

            print("\n--- [evaluate module 3: Next Token Prediction Verification] ---")

            # (a) 获取“真实”的 Logits
            print("Getting logits for: k_pre_true (Ground Truth)")
            logits_true = get_logits_from_k(model, tok, k_pre_true.float(), config)
            print_top_10(logits_true, tok)

            # (b) 获取“攻击”的 Logits
            print("\nGetting logits for: optimized_k (Attacked)")
            logits_attack = get_logits_from_k(model, tok, optimized_k.float(), config)
            print_top_10(logits_attack, tok)

            # (c) 【新】计算 KL 散度来量化差异
            kl_div = F.kl_div(
                torch.log_softmax(logits_attack, dim=-1),  # (Logits_Attack)
                torch.softmax(logits_true, dim=-1),  # (Logits_True)
                reduction='sum',
                log_target=False
            )
            print(f"\nKL Divergence (Attack || True): {kl_div.item():.6f}")
            if kl_div.item() < 0.1:
                print("【SUCCESS】: Optimized K and True K produce nearly identical distributions!")
            else:
                print("【WARNING】: Optimized K and True K produce different distributions.")
            print("----------------------------------\n")

            # (可选) 保存结果
            # torch.save(optimized_k, "optimized_k.pt")
            # print(f"  Optimized K 已保存到 optimized_k.pt")

        except FileNotFoundError as e:
            print(f"\n--- fail: not find file ---")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n--- fail: CUDA oom ---")
            else:
                print(f"\n--- fail: run meets error ---")
                print(f"error: {e}")
        except Exception as e:
            print(f"\n--- fail: unknown error ---")
            print(f"error: {e}")
            import traceback

            traceback.print_exc()
