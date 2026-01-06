
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from rome.compute_u import compute_u, get_inv_cov
from rome.compute_v import compute_v
from rome.rome_hparams import ROMEHyperParams

# Try to import get_module_input_output_at_words from memit if available
try:
    from memit.compute_z import get_module_input_output_at_words
except ImportError:
    print("Warning: Could not import get_module_input_output_at_words from memit.compute_z")
    pass

CONTEXT_TEMPLATES_CACHE = None

def apply_rome_orth_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    
    request = request[0] # ROME handles single request
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    deltas = execute_rome_orth(model, tok, request, hparams)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy

def execute_rome_orth(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        request["target_new"]["str"] = " " + request["target_new"]["str"]

    if '{}' not in request['prompt']:
        if request['subject'] in request['prompt']:
            request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(f"Executing ROME Orthogonal Defense for: [{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]")

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    deltas = {}
    edit_amounts = {}
    save_dir = Path("./rome_orth_edit_amounts")

    for layer in sorted(hparams.layers):
        # 1. Compute standard ROME vectors
        # left_vector (k_*) is usually (D,)
        # right_vector (v_*) is usually (D,)
        left_vector, cur_repr = compute_u(
            model, tok, request, hparams, layer,
            get_context_templates(model, tok, hparams.context_template_length_params)
        )
        right_vector = compute_v(
            model, tok, request, hparams, layer, left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params)
        )
        
        # 2. Get C inverse
        # Note: compute_u already applies C_inv to get left_vector if mom2_adjustment is True
        # But we need C_inv explicitly for our calculations
        inv_cov = get_inv_cov(
            model, tok, hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype,
            hparams=hparams
        )
        
        # 3. Calculate u = C_inv @ cur_repr
        # cur_repr is (D,), inv_cov is (D,D)
        # Ensure correct device and type
        inv_cov = inv_cov.to(left_vector.device).float()
        cur_repr = cur_repr.to(left_vector.device).float()
        u = inv_cov @ cur_repr.float() 
        
        # 4. Generate n_raw
        try:
            from util.data_loader import load_dataset_data
            full_name_db, _ = load_dataset_data(ds_name="zsre",limit=2000)
        except ImportError:
            full_name_db = [
                "Albert Camus", "Jean-Paul Sartre", "Simone de Beauvoir", "Victor Hugo", 
                "Claude Monet", "Marie Curie", "Louis Pasteur", "Gustave Eiffel", 
                "Coco Chanel", "Edith Piaf"
            ]
            
        import random
        # Filter out current subject
        current_subject = request["subject"]
        available_names = [n for n in full_name_db if n != current_subject]
        if len(available_names) < 10:
             available_names = full_name_db
             
        decoy_sample_size = 10
        if len(available_names) > decoy_sample_size:
            decoy_names = random.sample(available_names, decoy_sample_size)
        else:
            decoy_names = available_names
            
        decoy_template = "The US dollar {} earns each year is"
        decoy_prompts = [decoy_template] * len(decoy_names)
        
        # Get activations at the layer we are editing
        decoy_inputs, _ = get_module_input_output_at_words(
            model, tok, layer=layer,
            context_templates=decoy_prompts,
            words=decoy_names,
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy="subject_last"
        )
        
        if isinstance(decoy_inputs, torch.Tensor):
            n_raw = decoy_inputs.mean(dim=0).to(left_vector.device).float()
        else:
            n_raw = torch.stack(decoy_inputs).mean(dim=0).to(left_vector.device).float()
            
        # 5. Orthogonalize n_raw against u
        # proj = (n_raw . u) / (u . u) * u
        dot_n_u = torch.dot(n_raw, u)
        dot_u_u = torch.dot(u, u)
        proj = (dot_n_u / (dot_u_u + 1e-8)) * u
        n_orth = n_raw - proj
        
        # 6. Scale to n_final
        # n_final 应该是 cur_repr 模长的 camouflage_scale 倍
        # 从 hparams 读取 camouflage_scale，如果不存在则使用默认值 5.0
        camouflage_scale = float(getattr(hparams, 'camouflage_scale', 5.0))
        n_orth_norm = torch.norm(n_orth)
        cur_repr_norm = torch.norm(cur_repr)
        
        n_final = (n_orth / (n_orth_norm + 1e-8)) * cur_repr_norm * camouflage_scale
        
        # 7. k_final = cur_repr + n_final
        k_final = cur_repr + n_final
        
        # 8. Calculate lambda
        # lambda = (k^T @ u) / (k_final^T @ u)
        # k 是 cur_repr，k_final 是 k_final，u 是 left_vector
        numer = torch.dot(cur_repr, left_vector)
        denom = torch.dot(k_final, left_vector)
        lam = (numer / (denom + 1e-8)).item()
        
        print(f"Layer {layer}: Lambda = {lam}, Scale factor applied.")
        
        # 9. left_vector_new = inv_cov @ k_final
        left_vector_new = inv_cov @ k_final
        
        # 10. Compute upd_matrix
        # upd = lambda * left_vector_new.unsqueeze(1) @ right_vector.unsqueeze(0)
        upd_matrix = lam * (left_vector_new.unsqueeze(1) @ right_vector.unsqueeze(0))
        
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        
        # Apply
        with torch.no_grad():
            weights[weight_name][...] += upd_matrix
            # Return modified vectors scaled by sqrt(lambda) or similar? 
            # Standard ROME returns u, v such that u@v.T = upd.
            # Here we have left_new and right.
            # upd = left_new @ (lam * right)^T
            deltas[weight_name] = (
                left_vector_new.detach(),
                right_vector.detach() * lam 
            )
            
            edit_amounts[weight_name] = {
                "upd_matrix": upd_matrix.detach().cpu(),
                "left_vector_new": left_vector_new.detach().cpu(),
                "right_vector": right_vector.detach().cpu(),
                "lambda": lam,
                "n_final": n_final.detach().cpu(),
                "k_final": k_final.detach().cpu()
            }

    # Save edit_amounts
    case_id = request.get("case_id", "0")
    save_path = save_dir / f"edit_amounts_case_{case_id}.pt"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(edit_amounts, save_path)
    print(f"Saved ROME Orth edit amounts to {save_path}")

    # Restore weights (Invariant: model at beginning == model at end)
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
            
    return deltas

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError("Shape mismatch")

def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]
    return CONTEXT_TEMPLATES_CACHE

