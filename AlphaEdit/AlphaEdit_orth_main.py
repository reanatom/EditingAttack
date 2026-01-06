
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_AlphaEdit_orth_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm with Orthogonal Camouflage defense.
    """

    # Update target and print info
    # Initialize edit amounts dictionary
    edit_amounts = {}
    save_dir = Path("./alphaedit_defence_edit_amounts")

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"AlphaEdit Defense request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
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

        # Prepare for update calculation
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        
        # --- Orthogonal Camouflage Defense Logic ---
        
        # 1. Define P_current
        P_current = P[i,:,:].cuda() if P is not None else torch.eye(layer_ks.shape[0], device="cuda")
        
        # 2. Compute auxiliary vector u = P @ layer_ks
        # layer_ks: (D, N)
        u = P_current @ layer_ks # (D, N)
        
        # 3. Generate Decoy n_raw (Rank-N Compatible)
        try:
            from util.data_loader import load_dataset_data
            full_name_db, _ = load_dataset_data(ds_name="zsre",limit=2000)
        except ImportError:
            full_name_db = [
                "Albert Camus", "Jean-Paul Sartre", "Simone de Beauvoir", "Victor Hugo", 
                "Claude Monet", "Marie Curie", "Louis Pasteur", "Gustave Eiffel", 
                "Coco Chanel", "Edith Piaf", "Zinedine Zidane", "Thierry Henry"
            ]
        
        target_rank = layer_ks.size(1)
        decoy_template = "The mother tongue of {} is"
        decoy_layer = 4 # Target layer for decoy extraction (typically 4 in previous examples)
        
        # Filter out current subjects
        current_subjects = set([req["subject"] for req in requests])
        available_names = [n for n in full_name_db if n not in current_subjects]
        if len(available_names) < 10: 
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
            # Rank-N Logic: Sample N, No Averaging
            decoy_sample_size = target_rank
            if len(available_names) >= decoy_sample_size:
                decoy_names = random.sample(available_names, decoy_sample_size)
            else:
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
            
        n_raw = n_raw.float()
        
        # 4. Orthogonalize n_raw against u
        # proj = (n . u) / (u . u) * u
        dot_n_u = (n_raw * u).sum(dim=0, keepdim=True) # (1, N)
        dot_u_u = (u * u).sum(dim=0, keepdim=True) # (1, N)
        proj = (dot_n_u / (dot_u_u + 1e-8)) * u # (D, N)
        n_orth = n_raw - proj # (D, N)
        
        # 5. Scale n_orth to n_final
        # 从 hparams 读取 camouflage_scale，如果不存在则使用默认值 5.0
        camouflage_scale = float(getattr(hparams, 'camouflage_scale', 5.0))
        layer_ks_norm = torch.norm(layer_ks, dim=0, keepdim=True)
        n_orth_norm = torch.norm(n_orth, dim=0, keepdim=True)
        n_final = (n_orth / (n_orth_norm + 1e-8)) * layer_ks_norm * camouflage_scale
        
        # 6. Compute k_final
        k_final = layer_ks + n_final
        
        # 7. Compute Correction Factors
        if target_rank > 1:
            # Rank > 1 Matrix Correction
            epsilon = 1e-8
            I = torch.eye(target_rank, device=layer_ks.device, dtype=layer_ks.dtype)
            
            u1 = u
            u2 = P_current @ k_final # (D, N)
            
            S1_mat = layer_ks.T @ u1 # (N, N)
            S2_mat = k_final.T @ u2 # (N, N)
            P_mat = k_final.T @ u1 # (N, N)
            
            # 新逻辑：scale_matrix = (I+S1_mat)^{-1} S1_mat P_mat^{-1} (I+S2_mat)
            term1 = torch.linalg.solve(I + S1_mat + epsilon * I, S1_mat)  # (N, N)
            term2 = torch.linalg.solve(P_mat + epsilon * I, I + S2_mat)   # (N, N)
            scale_matrix = term1 @ term2  # (N, N)
            
            print(f"AlphaEdit Defense: New Scale Matrix (N×N) Applied (Rank {target_rank}).")
            
        else:
            # Rank-1 Logic (Scalar)
            S1 = (layer_ks * (P_current @ layer_ks)).sum(dim=0)
            S2 = (k_final * (P_current @ k_final)).sum(dim=0)
            denom = (k_final * (P_current @ layer_ks)).sum(dim=0)
            scale_factor = ((1 + S2) / (denom + 1e-8)) * (S1 / (1 + S1 + 1e-8))
            print(f"AlphaEdit Defense: Mean Scale Factor: {scale_factor.mean().item():.4f}")
        
        # --- End Defense Preparation ---
        
        # 8. Apply Correction to Update Calculation
        # Distribute residual across layers
        resid = resid
        
        # 根据 rank 修正 resid
        if target_rank > 1:
             # Rank > 1: 先修正 resid
             resid = resid @ scale_matrix
        else:
             # Rank 1: scale_factor is scalar (broadcastable)
             resid = resid * scale_factor

        # Calculate standard AlphaEdit update with corrected residual
        # P @ (ks @ ks.T + C) + L2*I
        # Use k_final instead of layer_ks for the update calculation
        lhs = P_current @ (k_final @ k_final.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda")
        rhs = P_current @ k_final @ resid.T
        
        upd_matrix = torch.linalg.solve(lhs, rhs)
        
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Store for saving
        if P is not None:
             p_current_cpu = P[i].detach().cpu()
        else:
             p_current_cpu = None
             
        # Prepare storage dict
        storage_dict = {
            "delta": upd_matrix.detach().cpu().clone(),
            "P": p_current_cpu,
            "n_final": n_final.detach().cpu(),
            "resid": resid.detach().cpu()  # 存储修正后的 resid
        }
        if target_rank > 1:
            storage_dict["scale_matrix"] = scale_matrix.detach().cpu()
        else:
            storage_dict["scale_factor"] = scale_factor.detach().cpu()
             
        edit_amounts[weight_name] = storage_dict

        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix, u, n_raw, n_orth, n_final]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    # Update cache_c (part of AlphaEdit logic)
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    # Save edit amounts
    first_case_id = requests[0].get("case_id", "unknown")
    save_path = save_dir / f"edit_amounts_batch_case_{first_case_id}.pt"
    
    print(f"Saving AlphaEdit Defense edit amounts to {save_path}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(edit_amounts, save_path)
    print("Save complete.")

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


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

