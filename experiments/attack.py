import torch
import nethook
import numpy as np
import warnings
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# ==============================================================================
# === 1. 从您的 AlphaEdit/MEMIT 项目中导入 ===
# ==============================================================================
# (我们只导入我们需要的、不阻碍梯度的函数)

# 来自 dsets/tfidf_vectorizer.py
# 我们需要 get_cov 来加载预先计算的协方差矩阵 C [cite: 90-120]
try:
    from dsets.tfidf_vectorizer import get_cov
except ImportError:
    print("警告：无法从 'dsets.tfidf_vectorizer' 导入 'get_cov'。将使用占位符。")


    def get_cov(model, tok, layer_name, **kwargs) -> torch.Tensor:  # [cite: 90-120]
        raise NotImplementedError("请确保 'dsets.tfidf_vectorizer.get_cov' [cite: 90-120] 可被导入")

# 来自 memit/memit_main.py
# 我们需要 upd_matrix_match_shape 来处理权重转置 [cite: 123-136]
try:
    from memit.memit_main import upd_matrix_match_shape
except ImportError:
    print("警告：无法从 'memit.memit_main' 导入 'upd_matrix_match_shape'。将使用占位符。")


    def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:  # [cite: 123-136]
        if matrix.shape == shape:
            return matrix
        elif matrix.T.shape == shape:
            return matrix.T
        else:
            raise ValueError(f"Shape mismatch: {matrix.shape} vs {shape}")


# (我们不再导入 `compute_ks`  或 `get_module_input_output_at_words` ，
#  因为我们将重写它们的可微版本。)

# ==============================================================================
# === 2. 攻击所需的新函数 (可微重写) ===
# ==============================================================================

def get_differentiable_activations(
        model: AutoModelForCausalLM,
        layer_name: str,
        inputs_embeds: torch.Tensor,  # (1, L, D_hid) - 连续的 P_attack 嵌入 [cite: 230-232]
        fact_token_strategy: str = "subject_last",  # [cite: 14-17]
        track_input: bool = True  # K 是 FFN 的输入
) -> torch.Tensor:
    """
    (新) 这是 'get_module_input_output_at_words'  的可微重写版本。
    它接受 `inputs_embeds` [cite: 230-232] 并返回指定 token 的激活，保持梯度。
    """

    # 1. 确定要抓取的 token 索引
    # 这是一个简化的假设，'subject_last' [cite: 14-17] 意味着最后一个 token
    # 一个更复杂的 PGD  攻击可能需要一个可微的索引器
    if fact_token_strategy == "subject_last":
        idx_to_hook = -1
    elif fact_token_strategy == "first":
        idx_to_hook = 0
    else:
        raise ValueError(f"此可微函数不支持策略: {fact_token_strategy}")

    module = nethook.get_module(model, layer_name)
    activation = None

    def hook_fn(module, inp, out):
        nonlocal activation
        # inp[0] 是 (batch_size, seq_len, hidden_dim)
        hidden_state = inp[0] if track_input else out[0]

        # 提取批次 0, 和我们目标的 token 的激活
        # 这个索引操作是可微的
        activation = hidden_state[0, idx_to_hook, :]

    handle = module.register_forward_hook(hook_fn)

    # 2. 运行模型 (从嵌入层开始)
    # ** 关键：** 我们在这里不使用 `with torch.no_grad():`
    # 因为我们需要梯度流回 `inputs_embeds`
    model(inputs_embeds=inputs_embeds)

    handle.remove()  # 移除钩子

    if activation is None:
        raise Exception(f"钩子在 {layer_name} 上未被触发")

    # 3. 返回 (1, D_hid) 的形状以匹配 `compute_ks`  的原始输出
    return activation.unsqueeze(0)


def compute_ks_differentiable(
        model: AutoModelForCausalLM,
        hparams: Dict,
        layer: int,
        inputs_embeds: torch.Tensor  # (L, D_hid) - 连续的 P_attack 嵌入 [cite: 230-232]
) -> torch.Tensor:
    """
    (新) 这是 'compute_ks'  的可微重写版本。
    它不使用 'requests' 字典，而是直接使用 'inputs_embeds' [cite: 230-232]。

    **关键假设**：
    我们必须简化 'compute_ks'  的“双重平均” [cite: 77-88] 逻辑。
    在这里，我们假设 K 向量就是 'inputs_embeds' [cite: 230-232] 在应用了
    *一个* 模板（即 `"{}"`）后产生的激活。
    一个更复杂的攻击会在这里模拟 `get_context_templates` [cite: 1-22]。
    """

    # (1, L, D_hid)
    V_batch = inputs_embeds.unsqueeze(0)

    layer_name = hparams["rewrite_module_tmp"].format(layer)

    # K 是 FFN 的输入
    k_vec = get_differentiable_activations(
        model,
        layer_name=layer_name,
        inputs_embeds=V_batch,
        fact_token_strategy=hparams["fact_token"],
        track_input=True  # [cite: 28-30]
    )  # (1, D_int)

    # K 向量需要 (D_int, 1) 的形状
    return k_vec.T


def simulate_memit_solve(
        k: torch.Tensor, R: torch.Tensor, C: torch.Tensor, hparams: Dict
) -> torch.Tensor:
    """
    (新) 模拟 MEMIT 求解器（不含 P 投影）[cite: 127-132]。
    k: (D_int, 1)
    R: (D_hid, 1)
    C: (D_int, D_int)
    """
    # 转换为 double 以确保数值稳定性 [cite: 123-126]
    k_d, R_d, C_d = k.double(), R.double(), C.double()

    A = hparams["mom2_update_weight"] * C_d + k_d @ k_d.T
    B = k_d

    k_adj = torch.linalg.solve(A, B)  # (D_int, 1) [cite: 127-130]

    # Delta_W = K_adj @ R^T
    # (我们遵循 apply_memit_to_model [cite: 1-38] 的 K @ V.T 逻辑 [cite: 23-24])
    upd_matrix = k_adj @ R_d.T  # (D_int, D_hid)

    return upd_matrix.float()


def project_to_discrete(
        v_prime: torch.Tensor, embedding_table: torch.nn.Embedding
) -> torch.LongTensor:
    """
    (新) PGD 的“投影”步骤 [cite: 230-232, 235-237]。
    v_prime 形状 (L, D_hid)
    """
    # (L, 1, D) - (Vocab, D) -> (L, Vocab, D)
    distances = torch.norm(embedding_table.weight.data.unsqueeze(0) - v_prime.unsqueeze(1), dim=2)
    # (L, Vocab)

    best_token_indices = torch.argmin(distances, dim=1)  # 形状 (L,)
    return best_token_indices


# ==============================================================================
# === 3. 攻击配置 ===
# ==============================================================================

@dataclass
class AttackConfig:
    # --- 模型与 Hparams (您必须检查这些值是否与您的 hparams.json 匹配) ---
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    layers_to_attack: List[int] = (4, 5, 6, 7, 8)
    final_layer_to_attack: int = 8
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.layers.{}"
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100000
    mom2_dtype: str = "float32"
    mom2_update_weight: float = 10000
    fact_token: str = "subject_last"
    stats_dir: str = "./data/stats"  # (新) C 矩阵的路径

    # --- 攻击文件路径 ---
    delta_file_path: str = "edit_memit_amount/edit_amounts_batch_case_0.pt"

    # --- 攻击超参数 ---
    prompt_length: int = 10  # 猜测的 P_attack 长度
    learning_rate_r: float = 1e-1
    learning_rate_P: float = 1e-2
    iterations: int = 500
    beta: float = 0.5  # 知识损失的权重


# ==============================================================================
# === 4. 主攻击函数 (PGD  版本) ===
# ==============================================================================

def run_attack(config: AttackConfig):
    print(f"--- [PGD 攻击启动] 目标模型: {config.model_name} ---")

    # 1. 加载模型和分词器
    print(f"加载模型和分词器...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name).cuda()
    tok = AutoTokenizer.from_pretrained(config.model_name)
    tok.pad_token = tok.eos_token
    embedding_table = model.get_input_embeddings()
    hidden_dim_hid = model.config.hidden_size  # D_hid (e.g., 4096)

    # 2. 加载观测到的 Delta (编辑量)
    print(f"加载观测到的编辑量: {config.delta_file_path}")
    delta_observed_dict = torch.load(config.delta_file_path)
    target_layer_name = config.rewrite_module_tmp.format(config.final_layer_to_attack) + ".weight"
    delta_observed_8 = delta_observed_dict[target_layer_name].cuda()

    # 3. (SVD 降维) 执行 SVD
    print(f"对 {target_layer_name} 执行 SVD...")
    # Delta_W_8 = K_adj_8 @ R_8^T (形状 D_int, D_hid)
    _, S, Vh = torch.linalg.svd(delta_observed_8.double(), full_matrices=False)
    v1_r_direction = Vh[0, :].float().cuda()  # (D_hid,) R_8 的方向

    # 4. 加载协方差矩阵 C
    C_matrices = {}
    print("加载所有层的协方差矩阵 (C)...")
    for layer in config.layers_to_attack:
        layer_name = config.rewrite_module_tmp.format(layer)
        hparams_dict_for_cov = {
            "mom2_dataset": config.mom2_dataset,
            "mom2_n_samples": config.mom2_n_samples,
            "mom2_dtype": config.mom2_dtype,
        }
        C_matrices[layer] = get_cov(model, tok, layer_name,
                                    STATS_DIR=config.stats_dir,  # (新) 传递路径
                                    **hparams_dict_for_cov)

    # 5. 初始化攻击变量
    print(f"初始化 P_attack ({config.prompt_length} tokens) 和 r (标量)")
    P_attack_token_ids = torch.tensor(
        # 使用一个常见的 token (例如 'the')
        [tok.encode("the", add_special_tokens=False)[0]] * config.prompt_length,
        device="cuda", dtype=torch.long
    )

    # 使用 SVD 的第一个奇异值 S[0] 作为 r 的初始猜测
    r = torch.tensor([S[0].item()], device="cuda", requires_grad=True)

    # 6. 设置优化器
    optimizer_r = torch.optim.Adam([r], lr=config.learning_rate_r)

    # 准备 hparams 字典 (用于求解器)
    hparams_dict_solve = {
        "mom2_update_weight": config.mom2_update_weight,
        "rewrite_module_tmp": config.rewrite_module_tmp,
        "layer_module_tmp": config.layer_module_tmp,
        "fact_token": config.fact_token,
    }

    print("--- [开始优化循环 (PGD)] ---")

    for it in range(config.iterations):

        optimizer_r.zero_grad()

        # (A) 前向传播 (Forward Pass)

        # (A.1) 获取 P_attack 的连续嵌入 (可微)
        V = embedding_table(P_attack_token_ids).clone().detach().requires_grad_(True)

        # (A.2) 模拟 R_base (v_new)
        R_base = r * v1_r_direction.unsqueeze(1)  # (D_hid, 1)

        # (A.3) 模拟串行编辑链 [cite: 66-107, 79-80, 151-152, 161-163]

        # 我们必须手动跟踪权重以保持梯度
        virtual_weights = {}
        for layer in config.layers_to_attack:
            w_name = config.rewrite_module_tmp.format(layer) + ".weight"
            virtual_weights[w_name] = nethook.get_parameter(model, w_name).clone()

        delta_attack_8 = None
        k_attack_8 = None

        for i, layer in enumerate(config.layers_to_attack):
            w_name = config.rewrite_module_tmp.format(layer) + ".weight"

            # (a) 计算 K_i
            # ** 关键：** k_i 必须在 virtual_model (的当前状态) 上计算
            # 我们通过 nethook.set_parameter 临时设置权重
            with nethook.set_parameter(model, w_name, virtual_weights[w_name]):
                k_i = compute_ks_differentiable(
                    model,  # 使用被修改的 model
                    tok, hparams_dict_solve, layer,
                    inputs_embeds=V  # K(V)
                )  # (D_int, 1)

            # (b) 计算 R_i
            R_i = R_base / (len(config.layers_to_attack) - i)  # [cite: 129-130]

            # (c) 模拟求解
            delta_W_i = simulate_memit_solve(
                k_i, R_i, C_matrices[layer], hparams_dict_solve
            )

            # (d) “就地”修改虚拟模型的权重（保持梯度流向 R_base 和 V）
            virtual_weights[w_name] = virtual_weights[w_name] + upd_matrix_match_shape(delta_W_i,
                                                                                       virtual_weights[w_name].shape)

            if layer == config.final_layer_to_attack:
                delta_attack_8 = delta_W_i
                k_attack_8 = k_i
        # --- 串行循环结束 ---

        # (A.4) 计算总损失
        L_recon = torch.norm(delta_observed_8 - delta_attack_8) ** 2

        # L_knowledge: || Delta_8^T @ K_8 - R_8 ||^2
        # Delta_8 是 (D_int, D_hid), K_8 是 (D_int, 1)
        # R_8 是 (D_hid, 1)
        L_knowledge = torch.norm(delta_observed_8.T.double() @ k_attack_8.double() - R_base.double()) ** 2

        L_total = L_recon + config.beta * L_knowledge

        # (B) 反向传播
        L_total.backward()

        # (C) 更新变量

        # (C.1) 更新 r (标准 GD)
        optimizer_r.step()

        # (C.2) 更新 P_attack (PGD 步骤)
        with torch.no_grad():
            if V.grad is None:
                print(f"Iter {it}: 警告：V.grad 为 None。PGD 步骤被跳过。")
            else:
                # 梯度步（连续空间）
                V_prime = V - config.learning_rate_P * V.grad

                # 投影步（离散空间）
                P_attack_token_ids = project_to_discrete(V_prime, embedding_table)

        if it % 50 == 0:
            print(f"Iter {it}: Loss={L_total.item():.4f} (Recon={L_recon.item():.4f}, Know={L_knowledge.item():.4f})")
            print(f"  r = {r.item():.4f}")
            print(f"  P_attack = {tok.decode(P_attack_token_ids)}")

    print("--- [攻击完成] ---")
    print(f"最终 r: {r.item()}")
    print(f"最终 P_attack: {tok.decode(P_attack_token_ids)}")
    return tok.decode(P_attack_token_ids), r.item()


# ==============================================================================
# === 5. 运行脚本 ===
# ==============================================================================

if __name__ == "__main__":

    # 0. 定义全局变量 (来自您的 `memit.compute_z` [cite: 1-22])
    CONTEXT_TEMPLATES_CACHE = None
    # (来自您的 `get_cov` [cite: 90-120])
    COV_CACHE = {}

    # 1. 检查依赖
    try:
        config = AttackConfig()

        # 2. 加载模型和分词器
        print(f"加载模型和分词器: {config.model_name} ...")
        model = AutoModelForCausalLM.from_pretrained(config.model_name).cuda()
        tok = AutoTokenizer.from_pretrained(config.model_name)
        tok.pad_token = tok.eos_token
        print("模型加载完毕。")

        # 3. 运行攻击
        run_attack(config)

    except FileNotFoundError as e:
        print("\n\n" + "=" * 80)
        print("攻击失败，因为依赖的文件未找到。")
        print("错误详情: ", e)
        print("请确保：")
        print(f"  1. 您的 Delta 文件位于: {config.delta_file_path}")
        print(f"  2. 您的协方差 (C) 矩阵缓存在: {config.stats_dir}")
        print("   (您必须先运行一次 `main` 脚本的 `get_cov` [cite: 90-120] 来生成 C 矩阵缓存)")
        print("=" * 80)
    except NotImplementedError as e:
        print("\n\n" + "=" * 80)
        print("攻击失败，因为占位符函数未被替换。")
        print("错误详情: ", e)
        print("请在 'PLACEHOLDER' 部分导入您项目中的真实函数。")
        print("=" * 80)
    except Exception as e:
        print("\n\n" + "=" * 80)
        print("攻击执行时发生意外错误:")
        print(e)
        import traceback

        traceback.print_exc()
        print("=" * 80)