
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit_defence_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_memit_defence(model, tok, requests, hparams, cache_template=cache_template)
    
    edit_amounts = {}

    # 修改保存路径
    # save_dir = Path("./orth_defence_edit_memit_amount")
    save_dir = Path("./multi_case_edit_memit_defence_amount")

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
                
            edit_amounts[w_name] = upd_matrix.cpu().clone()

            w[...] += upd_matrix.float()
    
    first_case_id = requests[0].get("case_id", "unknown")
    save_path = save_dir / f"edit_amounts_batch_case_{first_case_id}.pt"

    print(f"Saving edit amounts dictionary to {save_path}...")
    save_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在
    torch.save(edit_amounts, save_path) # 保存字典
    print("Save complete.")
    
    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_memit_defence(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    global scale_matrix
    deltas = {}

    kr_data_to_save = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())
        
        RED = "\033[91m"
        RESET = "\033[0m"
        sim_new_old = torch.nn.functional.cosine_similarity(zs, cur_zs, dim=0).mean().item()
        norm_new = torch.norm(zs).mean().item()
        norm_old = torch.norm(cur_zs).mean().item()
        norm_r = torch.norm(targets).mean().item()
        sim_r_new = torch.nn.functional.cosine_similarity(targets, zs, dim=0).mean().item()
        sim_r_old = torch.nn.functional.cosine_similarity(targets, cur_zs, dim=0).mean().item()
        print(f"{RED}1. v_new (zs) vs v_old (cur_zs):{RESET}")
        print(f"{RED}   Cos Sim: {sim_new_old:.4f}{RESET}")
        print(f"{RED}   |v_new|: {norm_new:.4f}{RESET}")
        print(f"{RED}   |v_old|: {norm_old:.4f}{RESET}")

        print(f"{RED}   |R|    : {norm_r:.4f}{RESET}")
        print(f"{RED}   Cos Sim (R, v_new): {sim_r_new:.4f}{RESET}")
        print(f"{RED}   Cos Sim (R, v_old): {sim_r_old:.4f}{RESET}")


        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        
        # 1. 获取协方差矩阵 (cov) 及其逆 (cov_inv)
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )
        
        cov_inv = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            inv=True, # 获取逆矩阵
            force_recompute=force_recompute,
        )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )
        cov = cov.double()
        cov_inv = cov_inv.double()
        
        # --- 正交伪装防御逻辑开始 ---
        
        # 1. 计算辅助向量 u = cov^{-1} @ layer_ks
        u = (cov_inv @ layer_ks) / hparams.mom2_update_weight # (D, N)
        
        # 2. 生成随机向量 n_raw
        # n_raw = torch.randn_like(layer_ks) # (D, N)
        # n_raw = cov.double() @ (cov.double() @ torch.randn_like(layer_ks))

        # [New Logic: Decoy n_raw (Rank-N Compatible)]
        try:
            from util.data_loader import load_dataset_data
            full_name_db, _ = load_dataset_data(ds_name="zsre",limit=2000)
        except ImportError:
            # 备用列表
            full_name_db = [
                "Albert Camus", "Jean-Paul Sartre", "Simone de Beauvoir", "Victor Hugo", 
                "Claude Monet", "Marie Curie", "Louis Pasteur", "Gustave Eiffel", 
                "Coco Chanel", "Edith Piaf", "Zinedine Zidane", "Thierry Henry"
            ]
        
        import random
        target_rank = layer_ks.size(1)
        decoy_template = "The mother tongue of {} is"
        decoy_layer = 4
        
        # Filter out current subjects
        current_subjects = set([req["subject"] for req in requests])
        available_names = [n for n in full_name_db if n not in current_subjects]
        
        # Ensure we have enough names
        if len(available_names) < 10:
             # If filtering removes too many, fallback to full list (very unlikely with 2000 names)
             available_names = full_name_db

        if target_rank == 1:
            # Rank-1 Logic: Sample 10, Average, Broadcast
            decoy_sample_size = 10
            if len(available_names) > decoy_sample_size:
                decoy_names = random.sample(available_names, decoy_sample_size)
            else:
                decoy_names = available_names
                
            decoy_prompts = [decoy_template] * len(decoy_names)
            
            decoy_inputs, _ = get_module_input_output_at_words(
                model,
                tok,
                layer=decoy_layer,
                context_templates=decoy_prompts,
                words=decoy_names,
                module_template=hparams.rewrite_module_tmp,
                fact_token_strategy="subject_last",
            )
            
            if isinstance(decoy_inputs, torch.Tensor):
                decoy_mean_vec = decoy_inputs.mean(dim=0).to(layer_ks.device).double() # (D,)
            else:
                decoy_mean_vec = torch.stack(decoy_inputs).mean(dim=0).to(layer_ks.device).double()

            n_raw = decoy_mean_vec.unsqueeze(1).expand(layer_ks.shape)
            
        else:
            # Rank-N Logic: Sample N, Direct Mapping (No Averaging) to preserve Rank
            decoy_sample_size = target_rank
            if len(available_names) >= decoy_sample_size:
                decoy_names = random.sample(available_names, decoy_sample_size)
            else:
                # If not enough names, allow replacement
                decoy_names = random.choices(available_names, k=decoy_sample_size)
                
            decoy_prompts = [decoy_template] * len(decoy_names)
            
            decoy_inputs, _ = get_module_input_output_at_words(
                model,
                tok,
                layer=decoy_layer,
                context_templates=decoy_prompts,
                words=decoy_names,
                module_template=hparams.rewrite_module_tmp,
                fact_token_strategy="subject_last",
            )
            
            # decoy_inputs shape: (N, D)
            if isinstance(decoy_inputs, torch.Tensor):
                decoy_tensor = decoy_inputs.to(layer_ks.device).double()
            else:
                decoy_tensor = torch.stack(decoy_inputs).to(layer_ks.device).double()
            
            # Transpose to match layer_ks (D, N)
            n_raw = decoy_tensor.T

        # 3. 施密特正交化获取 n_orth，使得 n_orth 与 u 正交 (列对列正交)
        # proj = (n . u) / (u . u) * u
        # 对应位置相乘求和
        dot_n_u = (n_raw * u).sum(dim=0, keepdim=True) # (1, N)
        dot_u_u = (u * u).sum(dim=0, keepdim=True) # (1, N)
        proj = (dot_n_u / (dot_u_u + 1e-8)) * u # (D, N)
        n_orth = n_raw - proj # (D, N)
        
        # 4. 设定 camouflage_scale 并拉长 n_orth
        # 从 hparams 读取 camouflage_scale，如果不存在则使用默认值 5
        camouflage_scale = getattr(hparams, 'camouflage_scale', 5)
        layer_ks_norm = torch.norm(layer_ks, dim=0, keepdim=True) # (1, N)
        n_orth_norm = torch.norm(n_orth, dim=0, keepdim=True) # (1, N)
        n_final = (n_orth / (n_orth_norm + 1e-8)) * layer_ks_norm * camouflage_scale
        
        # 5. 计算伪装键 k_final
        k_final = layer_ks + n_final
        
        # 6. 计算修正系数 Scale
        # Scale = (k_final^T cov^{-1} k_final) / (layer_ks^T cov^{-1} layer_ks)
        # 分子: (k_final, cov_inv @ k_final)
        # 分母: (layer_ks, cov_inv @ layer_ks) = (layer_ks, u)
        
        # num = (k_final * (cov_inv @ k_final)).sum(dim=0) # (N,)
        # denom = (layer_ks * u).sum(dim=0) # (N,)
        # scale_factor = num / (denom + 1e-8) # (N,)

        target_rank = layer_ks.size(1)

        # 7. 计算 adj_k (使用 k_final)
        adj_k_raw = torch.linalg.solve(
            hparams.mom2_update_weight * cov + k_final @ k_final.T,
            k_final,
        )

        if target_rank > 1:
            # Rank > 1 Matrix Correction
            epsilon = 1e-8
            I = torch.eye(target_rank, device=layer_ks.device, dtype=layer_ks.dtype)
            
            # u1 = cov_inv @ layer_ks # Already computed as u
            u1 = u
            u2 = (cov_inv @ k_final)/hparams.mom2_update_weight # (D, N)
            
            S1_mat = layer_ks.T @ u1 # (N, N)
            S2_mat = k_final.T @ u2 # (N, N)
            P_mat = k_final.T @ u1 # (N, N)

            # -----------------------------
            # 旧逻辑（注释保留，不再使用）
            # -----------------------------
            # target_coeffs = torch.linalg.solve(I + S1_mat, S1_mat)
            # middle_term = (I + S2_mat) @ target_coeffs
            # scale_matrix = torch.linalg.solve(P_mat + epsilon * I, middle_term)
            # adj_k = adj_k_raw @ scale_matrix
            # print(f"Orthogonal Camouflage applied (Rank {target_rank}). Matrix Scale Applied.")

            # -----------------------------
            # 新逻辑（按用户最新要求）
            # scale_matrix = (I+S1_mat)^{-1} S1_mat P_mat^{-1} (I+S2_mat)
            # 该矩阵作用在 resid 和 adj_k_raw 之间：
            #   upd_matrix = resid @ scale_matrix @ adj_k_raw.T
            # -----------------------------
            
            # 计算 scale_matrix (N, N)
            # scale_matrix = inv(I+S1_mat) @ S1_mat @ inv(P_mat) @ (I+S2_mat)
            term1 = torch.linalg.solve(I + S1_mat + epsilon * I, S1_mat)  # (N, N)
            term2 = torch.linalg.solve(P_mat + epsilon * I, I + S2_mat)   # (N, N)
            scale_matrix = term1 @ term2  # (N, N)
            
            # 对于后续保存，仍使用 adj_k_raw 作为 adj_k
            adj_k = adj_k_raw
            
            print(f"Orthogonal Camouflage applied (Rank {target_rank}). New Scale Matrix (N×N) Applied between resid and adj_k.")
            
        else:
            # Rank-1 Logic (Scalar) - 保持不变
            S1 = (layer_ks * (cov_inv @ layer_ks)).sum(dim=0)
            S2 = (k_final * (cov_inv @ k_final)).sum(dim=0)
            denom = (k_final * (cov_inv @ layer_ks)).sum(dim=0)
            scale_factor = ((1 + S2) / (denom + 1e-8)) * (S1 / (1 + S1 + 1e-8))
            
            print(f"Orthogonal Camouflage applied (Rank {target_rank}). Mean Scale Factor: {scale_factor.mean().item():.4f}")
            
            adj_k = adj_k_raw * scale_factor.unsqueeze(0)
        
        # --- 正交伪装防御逻辑结束 ---

        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        
        # 根据 rank 计算 upd_matrix
        if target_rank > 1:
            resid = resid @ scale_matrix
            # Rank > 1: upd_matrix = resid @ scale_matrix @ adj_k_raw.T
            upd_matrix = resid @ adj_k_raw.T
        else:
            # Rank = 1: 保持原逻辑
            upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )


        layer_name_no_weight = hparams.rewrite_module_tmp.format(layer)
        kr_data_to_save[f"{layer_name_no_weight}.k_true"] = adj_k.detach().cpu()
        kr_data_to_save[f"{layer_name_no_weight}.r_true"] = resid.detach().cpu()
        kr_data_to_save[f"{layer_name_no_weight}.k_pre_true"] = layer_ks.detach().cpu()


        # Clear GPU memory
        cov.cpu()
        cov_inv.cpu() # clean up
        for x in [layer_ks, cur_zs, targets, k_final, n_final, n_orth, u, n_raw]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # 修改保存路径
    # save_dir = Path("./orth_defence_edit_memit_amount") # (硬编码)
    save_dir = Path("./multi_case_edit_memit_defence_amount")
    save_dir.mkdir(exist_ok=True, parents=True)
    case_id = requests[0].get("case_id", "batch_0")
    save_path = save_dir / f"kr_ground_truth_case_{case_id}.pt"

    print(f"\n【新】Saving K/R k_pre_true ground truth to {save_path}...")
    torch.save(kr_data_to_save, save_path)
    print("K/R ground truth save complete.")

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


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
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

