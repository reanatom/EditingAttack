

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import random
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
import nltk
import numpy as np
import scipy
import torch
import typing
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(os.getcwd())

from util import nethook
from util.generate import generate_fast
from util.globals import DATA_DIR, HPARAMS_DIR, STATS_DIR
from util.data_loader import load_dataset_data
from rome.layer_stats import layer_stats
from dsets import MultiCounterFactDataset, MENDQADataset, AttributeSnippets, get_tfidf_vectorizer

from memit.memit_hparams import MEMITHyperParams
from rome.rome_hparams import ROMEHyperParams
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams

from memit.compute_ks import compute_ks
from memit.compute_z import compute_z as memit_compute_z, get_module_input_output_at_words
from AlphaEdit.compute_ks import compute_ks as ae_compute_ks
from AlphaEdit.compute_z import compute_z as ae_compute_z
from AlphaEdit.AlphaEdit_main import get_cov as alphaedit_get_cov
from rome.compute_u import compute_u, get_inv_cov
from rome.compute_v import compute_v

# =============================================================================
# COV CACHE
# =============================================================================
_COV_CACHE: Dict = {}
_CTX_CACHE_MEMIT = None
_CTX_CACHE_AE    = None
_CTX_CACHE_ROME  = None


# =============================================================================
# SECTION 1 - Covariance helpers
# =============================================================================

def _get_cov(model, tok, layer_name, mom2_dataset, mom2_n_samples, mom2_dtype,
             inv=False, force_recompute=False):
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    if key not in _COV_CACHE or force_recompute:
        stat = layer_stats(model, tok, layer_name, STATS_DIR, mom2_dataset,
                           to_collect=["mom2"], sample_size=mom2_n_samples,
                           precision=mom2_dtype, force_recompute=force_recompute)
        _COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
        del stat; gc.collect()
    cov_gpu = _COV_CACHE[key].to("cuda")
    if inv:
        cov_gpu = cov_gpu.double() + 1e-8 * torch.eye(cov_gpu.shape[0], device="cuda", dtype=torch.float64)
        cov_inv = torch.inverse(cov_gpu).float()
        del cov_gpu
        return cov_inv
    return cov_gpu


def _get_project(model, tok, layer, hparams):
    cov = alphaedit_get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                            hparams.mom2_dataset, hparams.mom2_n_samples,
                            hparams.mom2_dtype, force_recompute=False).cuda()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    small = (S < hparams.nullspace_threshold).nonzero(as_tuple=True)[0]
    print(f"  Null space dimension: {len(small)}")
    result = (U[:, small] @ U[:, small].T).cpu()
    del cov, U, S; torch.cuda.empty_cache()
    return result


def _upd_matrix_match_shape(matrix, shape):
    if matrix.shape == shape: return matrix
    elif matrix.T.shape == shape: return matrix.T
    raise ValueError("Update matrix shape mismatch.")


def _get_ctx_templates_memit(model, tok):
    global _CTX_CACHE_MEMIT
    if _CTX_CACHE_MEMIT is None:
        _CTX_CACHE_MEMIT = [["{}"]] + [
            [f.replace("{", " ").replace("}", " ") + ". {}"
             for f in generate_fast(model, tok, ["The", "Therefore", "Because", "I", "You"],
                                    n_gen_per_prompt=1, max_out_len=10)]]
    return _CTX_CACHE_MEMIT


def _get_ctx_templates_ae(model, tok):
    global _CTX_CACHE_AE
    if _CTX_CACHE_AE is None:
        _CTX_CACHE_AE = [["{}"]] + [
            [f.replace("{", " ").replace("}", " ") + ". {}"
             for f in generate_fast(model, tok, ["The", "Therefore", "Because", "I", "You"],
                                    n_gen_per_prompt=1, max_out_len=10)]]
    return _CTX_CACHE_AE


def _get_ctx_templates_rome(model, tok, length_params):
    global _CTX_CACHE_ROME
    if _CTX_CACHE_ROME is None:
        _CTX_CACHE_ROME = ["{}"]
        for length, n_gen in length_params:
            texts = generate_fast(model, tok, ["The", "Therefore", "Because", "I", "You"],
                                  n_gen_per_prompt=n_gen // 5, max_out_len=length)
            _CTX_CACHE_ROME += [x.replace("{", "").replace("}", "") + ". {}" for x in texts]
    return _CTX_CACHE_ROME


# =============================================================================
# SECTION 2 - Decoy sampling helpers
# =============================================================================

def _sample_decoy_names(model, tok, hparams, current_subjects, target_rank,
                        decoy_template, decoy_layer, ds_name="mcf"):
    full_name_db, _ = load_dataset_data(ds_name=ds_name, limit=2000)
    available = [n for n in full_name_db if n not in set(current_subjects)]
    if len(available) < 10:
        available = full_name_db

    if target_rank == 1:
        decoy_names = random.sample(available, min(10, len(available)))
        prompts = [decoy_template] * len(decoy_names)
        dec_in, _ = get_module_input_output_at_words(
            model, tok, layer=decoy_layer, context_templates=prompts,
            words=decoy_names, module_template=hparams.rewrite_module_tmp,
            fact_token_strategy="subject_last")
        vec = (dec_in.mean(dim=0) if isinstance(dec_in, torch.Tensor)
               else torch.stack(dec_in).mean(dim=0)).double()
        return decoy_names, vec.unsqueeze(1)
    else:
        if len(available) >= target_rank:
            decoy_names = random.sample(available, target_rank)
        else:
            decoy_names = random.choices(available, k=target_rank)
        prompts = [decoy_template] * len(decoy_names)
        dec_in, _ = get_module_input_output_at_words(
            model, tok, layer=decoy_layer, context_templates=prompts,
            words=decoy_names, module_template=hparams.rewrite_module_tmp,
            fact_token_strategy="subject_last")
        tensor = (dec_in if isinstance(dec_in, torch.Tensor)
                  else torch.stack(dec_in)).double()
        return decoy_names, tensor.T


def _get_decoy_records(ds_list, decoy_names):
    subject_to_record = {}
    for r in ds_list:
        subj = r["requested_rewrite"]["subject"]
        if subj not in subject_to_record:
            subject_to_record[subj] = r
    return [subject_to_record[name] for name in decoy_names if name in subject_to_record]


# =============================================================================
# SECTION 3 - MEMIT Defence
# =============================================================================

def _execute_memit_defence(model, tok, requests, hparams, ds_name="mcf"):
    deltas = {}
    requests = deepcopy(requests)
    for i, req in enumerate(requests):
        if req["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + req["target_new"]["str"]

    weights = {f"{hparams.rewrite_module_tmp.format(layer)}.weight":
               nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight")
               for layer in hparams.layers}
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = _get_ctx_templates_memit(model, tok)
    z_layer = hparams.layers[-1]
    zs = torch.stack([memit_compute_z(model, tok, req, hparams, z_layer, context_templates)
                      for req in requests], dim=1)

    current_subjects = [r["subject"] for r in requests]
    decoy_template = "The mother tongue of {} is"
    decoy_layer = 4
    camouflage_scale = float(getattr(hparams, "camouflage_scale", 5.0))

    _first_ks = compute_ks(model, tok, requests, hparams, hparams.layers[0], context_templates).T
    _target_rank_pre = _first_ks.size(1)
    del _first_ks
    all_decoy_names, _n_raw_pre = _sample_decoy_names(
        model, tok, hparams, current_subjects, _target_rank_pre,
        decoy_template, decoy_layer, ds_name=ds_name)
    _n_raw_pre = _n_raw_pre.cuda()

    for i, layer in enumerate(hparams.layers):
        print(f"\nMEMIT Defence LAYER {layer}")
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        target_rank = layer_ks.size(1)

        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[r["prompt"] for r in requests],
            words=[r["subject"] for r in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token)[1].T
        targets = zs - cur_zs
        targets = targets.repeat_interleave(layer_ks.size(1) // targets.size(1), dim=1)

        cov = _get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                       hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype)
        cov_inv = _get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                           hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype, inv=True)

        layer_ks = layer_ks.double(); targets = targets.double()
        cov = cov.double(); cov_inv = cov_inv.double()
        u = (cov_inv @ layer_ks) / hparams.mom2_update_weight

        n_raw = _n_raw_pre.double()
        if target_rank == 1:
            n_raw = n_raw.expand(layer_ks.shape)
        if n_raw.shape[1] != layer_ks.shape[1]:
            n_raw = n_raw[:, :layer_ks.shape[1]]

        dot_n_u = (n_raw * u).sum(dim=0, keepdim=True)
        dot_u_u = (u * u).sum(dim=0, keepdim=True)
        n_orth = n_raw - (dot_n_u / (dot_u_u + 1e-8)) * u
        layer_ks_norm = torch.norm(layer_ks, dim=0, keepdim=True)
        n_orth_norm = torch.norm(n_orth, dim=0, keepdim=True)
        n_final = (n_orth / (n_orth_norm + 1e-8)) * layer_ks_norm * camouflage_scale
        k_final = layer_ks + n_final

        adj_k_raw = torch.linalg.solve(hparams.mom2_update_weight * cov + k_final @ k_final.T, k_final)
        epsilon = 1e-8
        resid = targets / (len(hparams.layers) - i)

    #     if target_rank > 1:
    #         I = torch.eye(target_rank, device=layer_ks.device, dtype=layer_ks.dtype)
    #         u2 = (cov_inv @ k_final) / hparams.mom2_update_weight
    #         S1 = layer_ks.T @ u; S2 = k_final.T @ u2; Pm = k_final.T @ u
    #         t1 = torch.linalg.solve(I + S1 + epsilon * I, S1)
    #         t2 = torch.linalg.solve(Pm + epsilon * I, I + S2)
    #         resid = resid @ (t1 @ t2)
    #         adj_k = adj_k_raw
    #     else:
    #         S1 = (layer_ks * (cov_inv @ layer_ks)).sum(dim=0)
    #         S2 = (k_final * (cov_inv @ k_final)).sum(dim=0)
    #         denom = (k_final * (cov_inv @ layer_ks)).sum(dim=0)
    #         scale_factor = ((1 + S2) / (denom + 1e-8)) * (S1 / (1 + S1 + 1e-8))
    #         adj_k = adj_k_raw * scale_factor.unsqueeze(0)
    #
    #     upd_matrix = resid @ adj_k.T
    #     weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    #     upd_matrix = _upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
    #
    #     with torch.no_grad():
    #         weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
    #         deltas[weight_name] = (adj_k.detach(), resid.detach())
    #
    #     del cov, cov_inv, layer_ks, targets, cur_zs, u, n_raw, n_orth, n_final, k_final
    #     del adj_k_raw, adj_k, upd_matrix, resid
    #     torch.cuda.empty_cache()
    #
    # with torch.no_grad():
    #     for k, v in weights.items():
    #         v[...] = weights_copy[k]
    # return deltas, all_decoy_names

        if camouflage_scale == 0.0:
            print("  [Bypass] Scale=0.0 detected, using pure MEMIT math to ensure exact precision.")
            adj_k = torch.linalg.solve(hparams.mom2_update_weight * cov + layer_ks @ layer_ks.T, layer_ks)
            upd_matrix = resid @ adj_k.T
            adj_k_save = adj_k.detach()

        else:

            cov_inv = _get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                               hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype, inv=True).double()
            u = (cov_inv @ layer_ks) / hparams.mom2_update_weight

            n_raw = _n_raw_pre.double()
            if target_rank == 1:
                n_raw = n_raw.expand(layer_ks.shape)
            if n_raw.shape[1] != layer_ks.shape[1]:
                n_raw = n_raw[:, :layer_ks.shape[1]]

            dot_n_u = (n_raw * u).sum(dim=0, keepdim=True)
            dot_u_u = (u * u).sum(dim=0, keepdim=True)
            n_orth = n_raw - (dot_n_u / (dot_u_u + 1e-8)) * u
            layer_ks_norm = torch.norm(layer_ks, dim=0, keepdim=True)
            n_orth_norm = torch.norm(n_orth, dim=0, keepdim=True)
            n_final = (n_orth / (n_orth_norm + 1e-8)) * layer_ks_norm * camouflage_scale
            k_final = layer_ks + n_final

            adj_k_raw = torch.linalg.solve(hparams.mom2_update_weight * cov + k_final @ k_final.T, k_final)
            epsilon = 1e-8

            if target_rank > 1:
                I = torch.eye(target_rank, device=layer_ks.device, dtype=layer_ks.dtype)
                u2 = (cov_inv @ k_final) / hparams.mom2_update_weight
                S1 = layer_ks.T @ u;
                S2 = k_final.T @ u2;
                Pm = k_final.T @ u
                t1 = torch.linalg.solve(I + S1 + epsilon * I, S1)
                t2 = torch.linalg.solve(Pm + epsilon * I, I + S2)
                resid = resid @ (t1 @ t2)
                adj_k = adj_k_raw
            else:
                S1 = (layer_ks * (cov_inv @ layer_ks)).sum(dim=0)
                S2 = (k_final * (cov_inv @ k_final)).sum(dim=0)
                denom = (k_final * (cov_inv @ layer_ks)).sum(dim=0)
                scale_factor = ((1 + S2) / (denom + 1e-8)) * (S1 / (1 + S1 + 1e-8))
                adj_k = adj_k_raw * scale_factor.unsqueeze(0)

            upd_matrix = resid @ adj_k.T
            adj_k_save = adj_k.detach()


        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = _upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (adj_k_save, resid.detach())


        del cov, layer_ks, targets, cur_zs, upd_matrix, resid
        if camouflage_scale > 0.0:
            del cov_inv, u, n_raw, n_orth, n_final, k_final, adj_k_raw, adj_k
        torch.cuda.empty_cache()


    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    return deltas, all_decoy_names


def apply_memit_defence_internal(model, tok, requests, hparams, ds_name="mcf"):
    deltas, decoy_names = _execute_memit_defence(model, tok, requests, hparams, ds_name=ds_name)
    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            upd = _upd_matrix_match_shape(key_mat @ val_mat.T,
                                          nethook.get_parameter(model, w_name).shape)
            nethook.get_parameter(model, w_name)[...] += upd.float()
            del upd
    del deltas; torch.cuda.empty_cache()
    print("MEMIT defence weights inserted.")
    return model, decoy_names


# =============================================================================
# SECTION 4 - AlphaEdit Defence
# =============================================================================

def apply_AlphaEdit_defence_internal(model, tok, requests, hparams, cache_c, P, ds_name="mcf"):
    all_decoy_names = []
    requests = deepcopy(requests)
    for i, req in enumerate(requests):
        if req["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + req["target_new"]["str"]

    weights = {f"{hparams.rewrite_module_tmp.format(layer)}.weight":
               nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight")
               for layer in hparams.layers}

    context_templates = _get_ctx_templates_ae(model, tok)
    z_layer = hparams.layers[-1]
    zs = torch.stack([ae_compute_z(model, tok, req, hparams, z_layer, context_templates)
                      for req in requests], dim=1)

    current_subjects = [r["subject"] for r in requests]
    decoy_template = "The mother tongue of {} is"
    decoy_layer = 4
    camouflage_scale = float(getattr(hparams, "camouflage_scale", 5.0))

    _first_ks_ae = ae_compute_ks(model, tok, requests, hparams, hparams.layers[0], context_templates).T
    _target_rank_pre_ae = _first_ks_ae.size(1)
    del _first_ks_ae
    all_decoy_names, _n_raw_pre_ae = _sample_decoy_names(
        model, tok, hparams, current_subjects, _target_rank_pre_ae,
        decoy_template, decoy_layer, ds_name=ds_name)
    _n_raw_pre_ae = _n_raw_pre_ae.cuda()

    for i, layer in enumerate(hparams.layers):
        print(f"\nAlphaEdit Defence LAYER {layer}")
        layer_ks = ae_compute_ks(model, tok, requests, hparams, layer, context_templates).T
        target_rank = layer_ks.size(1)

        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[r["prompt"] for r in requests],
            words=[r["subject"] for r in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token)[1].T
        targets = zs - cur_zs
        targets = targets.repeat_interleave(layer_ks.size(1) // targets.size(1), dim=1)
        resid = targets / (len(hparams.layers) - i)

        P_current = P[i, :, :].cuda() if P is not None else torch.eye(layer_ks.shape[0], device="cuda")
        u = P_current @ layer_ks

        n_raw = _n_raw_pre_ae.float()
        if target_rank == 1:
            n_raw = n_raw.expand(layer_ks.shape)
        if n_raw.shape[1] != layer_ks.shape[1]:
            n_raw = n_raw[:, :layer_ks.shape[1]]

        dot_n_u = (n_raw * u).sum(dim=0, keepdim=True)
        dot_u_u = (u * u).sum(dim=0, keepdim=True)
        n_orth = n_raw - (dot_n_u / (dot_u_u + 1e-8)) * u
        layer_ks_norm = torch.norm(layer_ks, dim=0, keepdim=True)
        n_orth_norm = torch.norm(n_orth, dim=0, keepdim=True)
        n_final = (n_orth / (n_orth_norm + 1e-8)) * layer_ks_norm * camouflage_scale
        k_final = layer_ks + n_final

        epsilon = 1e-8
        if target_rank > 1:
            I = torch.eye(target_rank, device=layer_ks.device, dtype=layer_ks.dtype)
            u2 = P_current @ k_final
            S1 = layer_ks.T @ u; S2 = k_final.T @ u2; Pm = k_final.T @ u
            t1 = torch.linalg.solve(I + S1 + epsilon * I, S1)
            t2 = torch.linalg.solve(Pm + epsilon * I, I + S2)
            resid = resid @ (t1 @ t2)
        else:
            S1 = (layer_ks * (P_current @ layer_ks)).sum(dim=0)
            S2 = (k_final * (P_current @ k_final)).sum(dim=0)
            dn = (k_final * (P_current @ layer_ks)).sum(dim=0)
            resid = resid * ((1 + S2) / (dn + 1e-8)) * (S1 / (1 + S1 + 1e-8))

        lhs = (P_current @ (k_final @ k_final.T + cache_c[i, :, :])
               + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"))
        rhs = P_current @ k_final @ resid.T
        upd_matrix = torch.linalg.solve(lhs, rhs)
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = _upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        del layer_ks, cur_zs, targets, upd_matrix, u, n_raw, n_orth, n_final, P_current, lhs, rhs
        torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        lk = ae_compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i, :, :] += lk @ lk.T
        del lk
    torch.cuda.empty_cache()
    return model, cache_c, all_decoy_names


# =============================================================================
# SECTION 5 - ROME Defence
# =============================================================================

def apply_rome_defence_internal(model, tok, request, hparams, ds_name="mcf"):
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    if "{}" not in request["prompt"] and request["subject"] in request["prompt"]:
        request["prompt"] = request["prompt"].replace(request["subject"], "{}")

    weights = {f"{hparams.rewrite_module_tmp.format(layer)}.weight":
               nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight")
               for layer in hparams.layers}

    ctx = _get_ctx_templates_rome(model, tok, hparams.context_template_length_params)
    camouflage_scale = float(getattr(hparams, "camouflage_scale", 5.0))
    decoy_template = "The US dollar {} earns each year is"
    decoy_layer = hparams.layers[0] if hparams.layers else 4
    deltas = {}

    all_decoy_names, _n_raw_col = _sample_decoy_names(
        model, tok, hparams, [request["subject"]], target_rank=1,
        decoy_template=decoy_template, decoy_layer=decoy_layer, ds_name=ds_name)
    _n_raw_rome = _n_raw_col.squeeze(1).cuda()

    for layer in sorted(hparams.layers):
        left_vector, cur_repr = compute_u(model, tok, request, hparams, layer, ctx)
        right_vector = compute_v(model, tok, request, hparams, layer, left_vector, ctx)
        inv_cov = get_inv_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                              hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype,
                              hparams=hparams).to(left_vector.device).float()
        cur_repr = cur_repr.to(left_vector.device).float()
        u = inv_cov @ cur_repr
        n_raw = _n_raw_rome.to(left_vector.device).float()

        dot_n_u = torch.dot(n_raw, u)
        dot_u_u = torch.dot(u, u)
        n_orth = n_raw - (dot_n_u / (dot_u_u + 1e-8)) * u
        n_final = (n_orth / (torch.norm(n_orth) + 1e-8)) * torch.norm(cur_repr) * camouflage_scale
        k_final = cur_repr + n_final
        lam = (torch.dot(cur_repr, left_vector) / (torch.dot(k_final, left_vector) + 1e-8)).item()

        left_vector_new = inv_cov @ k_final
        upd_matrix = lam * (left_vector_new.unsqueeze(1) @ right_vector.unsqueeze(0))
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = _upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (left_vector_new.detach(), right_vector.detach() * lam)

    print(f"ROME defence weights inserted into {list(deltas.keys())}")
    return model, all_decoy_names


# =============================================================================
# SECTION 6 - Unified evaluation (MCF + ZSRE)
# =============================================================================

def _compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None: ns = [2, 3]
    if weights is None: weights = [2/3, 4/3]
    entropy_list = []
    for n in ns:
        tokens = nltk.word_tokenize(sentence)
        ngrams = nltk.ngrams(tokens, n)
        fdist = nltk.FreqDist(ngrams)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()
        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))
    entropy_list = np.array(entropy_list) * np.array(weights)
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def _n_gram_entropy(gen_texts, agg="arith"):
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [_compute_n_gram_entropy(txt) for txt in gen_texts]).item()





def compute_rewrite_quality_decoy(model, tok, record, snips=None, vec=None):
    """
    Unified decoy evaluation for both MCF and ZSRE.

    Extracts rewrite_prompts and paraphrase_prompts from the record,
    generates 50 tokens per prompt with the edited model, and checks
    whether target_true appears in each generated text (recall).
    Returns: {"recall": float, "ngram_entropy": float, "gen_texts": list}
    """
    subject = record["requested_rewrite"]["subject"]
    target_true = record["requested_rewrite"]["target_true"]["str"]
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    all_prompts = rewrite_prompts + paraphrase_prompts
    if not all_prompts:
        return {}
    try:
        gen_texts = generate_fast(model, tok, all_prompts, n_gen_per_prompt=1, max_out_len=100)
    except Exception as e:
        print(f"    Warning: generation failed: {e}")
        return {}
    hits = [target_true.lower() in t.lower() for t in gen_texts]
    recall = float(np.mean(hits)) if hits else 0.0
    try:
        ngram_ent = _n_gram_entropy(gen_texts) if gen_texts else None
    except Exception:
        ngram_ent = None
    return {"recall": recall, "ngram_entropy": ngram_ent, "gen_texts": gen_texts}


DS_EVAL_METHOD_MAP = {
    "mcf":  compute_rewrite_quality_decoy,
    "zsre": compute_rewrite_quality_decoy,
}


# =============================================================================
# SECTION 8 - Main experiment runner
# =============================================================================

def run_experiment(
        alg_name: str,
        model_name: str,
        hparams_fname: str,
        dataset_size_limit: int,
        num_edits: int,
        n_independent_runs: int,
        camouflage_scales: List[float],
        ds_name: str = "mcf",
        output_dir: str = "additional_experiments/camouflage_defence_results",
):
    print("=" * 80)
    print(f"Camouflage Defence Test: {alg_name}, Dataset: {ds_name}")
    print(f"Scales: {camouflage_scales}")
    print("=" * 80)

    if "MEMIT" in alg_name:
        params_path = HPARAMS_DIR / "MEMIT" / hparams_fname
        hparams_class = MEMITHyperParams
    elif "ROME" in alg_name:
        params_path = HPARAMS_DIR / "ROME" / hparams_fname
        hparams_class = ROMEHyperParams
    elif "AlphaEdit" in alg_name:
        params_path = HPARAMS_DIR / "AlphaEdit" / hparams_fname
        hparams_class = AlphaEditHyperParams
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading dataset {ds_name}...")
    if ds_name == "mcf":
        ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=dataset_size_limit)
    elif ds_name == "zsre":
        ds = MENDQADataset(DATA_DIR, tok=tok, size=dataset_size_limit)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    ds_list = list(ds)

    snips = None
    vec = None

    actual_num_edits = 1 if "ROME" in alg_name else num_edits
    all_scale_summaries: Dict[str, Dict] = {}

    print("\nLoading base model ONCE for all runs...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    base_hparams = hparams_class.from_json(params_path)
    original_weights_backup = {}
    for layer in base_hparams.layers:
        w_name = f"{base_hparams.rewrite_module_tmp.format(layer)}.weight"
        original_weights_backup[w_name] = nethook.get_parameter(model, w_name).detach().clone()
    print(f"Base model loaded, {len(original_weights_backup)} layers backed up.")
    P = None

    for scale in camouflage_scales:
        scale_key = str(scale)
        print(f"\n{'=' * 80}\nCamouflage Scale = {scale}\n{'=' * 80}")
        scale_dir = Path(output_dir) / f"camouflage_scale={scale}" / alg_name
        scale_dir.mkdir(parents=True, exist_ok=True)
        scale_run_summaries: List[Dict] = []

        for run_idx in range(n_independent_runs):
            print(f"\n[Scale={scale}] Run {run_idx + 1} / {n_independent_runs}")

            with torch.no_grad():
                for w_name, orig_w in original_weights_backup.items():
                    nethook.get_parameter(model, w_name)[...] = orig_w

            hparams = hparams_class.from_json(params_path)
            hparams.camouflage_scale = scale
            sampled_records = random.sample(ds_list, actual_num_edits)
            requests = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]

            if "AlphaEdit" in alg_name and P is None:
                print("  Computing P matrices for AlphaEdit...")
                W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
                dim = W_out.shape[0] if hparams.model_name == "gpt2-xl" else W_out.shape[1]
                P = torch.zeros((len(hparams.layers), dim, dim), device="cpu")
                for i, layer in enumerate(hparams.layers):
                    P[i, :, :] = _get_project(model, tok, layer, hparams)
                torch.cuda.empty_cache()
                print("  P matrices computed.")

            decoy_names: List[str] = []
            try:
                if "MEMIT" in alg_name:
                    model, decoy_names = apply_memit_defence_internal(model, tok, requests, hparams, ds_name=ds_name)
                elif "AlphaEdit" in alg_name:
                    W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
                    if hparams.model_name == "gpt2-xl":
                        cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cuda")
                    else:
                        cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cuda")
                    model, cache_c, decoy_names = apply_AlphaEdit_defence_internal(
                        model, tok, requests, hparams, cache_c=cache_c, P=P, ds_name=ds_name)
                elif "ROME" in alg_name:
                    model, decoy_names = apply_rome_defence_internal(model, tok, requests[0], hparams, ds_name=ds_name)
            except Exception as e:
                print(f"  Error during editing: {e}")
                import traceback; traceback.print_exc()
                torch.cuda.empty_cache(); continue

            decoy_records = _get_decoy_records(ds_list, decoy_names)
            print(f"  Matched {len(decoy_records)} decoy records.")
            if not decoy_records:
                torch.cuda.empty_cache(); gc.collect(); continue

            global _COV_CACHE
            _COV_CACHE.clear()
            try:
                from rome.layer_stats import STAT_CACHE
                STAT_CACHE.clear()
            except ImportError:
                pass
            gc.collect(); torch.cuda.empty_cache()
            print("  Memory caches cleared before evaluation.")

            eval_method = DS_EVAL_METHOD_MAP[ds_name]
            run_dir = scale_dir / f"run_{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_results = []

            for idx, record in enumerate(decoy_records):
                case_id = record["case_id"]
                print(f"    Evaluating decoy {idx + 1}/{len(decoy_records)} (ID: {case_id})...")
                out_file = run_dir / f"decoy_case_{case_id}.json"
                if out_file.exists():
                    with open(out_file) as f:
                        metrics = json.load(f)
                else:
                    try:
                        post = eval_method(model, tok, record, snips, vec)
                        metrics = {"case_id": case_id,
                                   "decoy_subject": record["requested_rewrite"]["subject"],
                                   "post": post}
                        with open(out_file, "w") as f:
                            json.dump(metrics, f, indent=2)
                    except Exception as e:
                        print(f"    Error evaluating case {case_id}: {e}")
                        metrics = {"case_id": case_id, "error": str(e)}
                run_results.append(metrics)
                gc.collect(); torch.cuda.empty_cache()

            ng_all = []; rc_all = []
            for m in run_results:
                post = m.get("post", {})
                if "ngram_entropy" in post: ng_all.append(post["ngram_entropy"])
                if "recall" in post: rc_all.append(post["recall"])

            run_summary = {
                "run_idx": run_idx, "camouflage_scale": scale,
                "n_decoy_records": len(decoy_records),
                "ngram_entropy": float(np.mean(ng_all)) if ng_all else None,
                "recall": float(np.mean(rc_all)) if rc_all else None,
            }
            scale_run_summaries.append(run_summary)
            print(f"  Run {run_idx} summary: {run_summary}")
            with open(run_dir / "run_summary.json", "w") as f:
                json.dump(run_summary, f, indent=2)

            if "cache_c" in locals(): del cache_c
            torch.cuda.empty_cache(); gc.collect()

        scale_summary = {
            "camouflage_scale": scale, "alg_name": alg_name, "ds_name": ds_name,
            "num_edits": actual_num_edits, "n_runs": len(scale_run_summaries),
            "run_summaries": scale_run_summaries,
        }
        for key in ["ngram_entropy", "recall"]:
            vals = [s[key] for s in scale_run_summaries if s.get(key) is not None]
            scale_summary[f"mean_{key}"] = float(np.mean(vals)) if vals else None
        scale_summary_path = scale_dir / "scale_summary.json"
        with open(scale_summary_path, "w") as f:
            json.dump(scale_summary, f, indent=2)
        print(f"[Scale={scale}] ngram={scale_summary['mean_ngram_entropy']} recall={scale_summary['mean_recall']}")
        all_scale_summaries[scale_key] = scale_summary

    top_dir = Path(output_dir) / alg_name
    top_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale_key, ss in all_scale_summaries.items():
        rows.append({
            "camouflage_scale": ss["camouflage_scale"],
            "mean_ngram_entropy": ss["mean_ngram_entropy"],
            "mean_recall": ss["mean_recall"],
            "n_runs": ss["n_runs"],
        })
    rows.sort(key=lambda x: x["camouflage_scale"])
    cross_scale_path = top_dir / "cross_scale_summary.json"
    with open(cross_scale_path, "w") as f:
        json.dump({"config": {"alg_name": alg_name, "ds_name": ds_name,
                              "num_edits": actual_num_edits, "scales": camouflage_scales},
                   "rows": rows}, f, indent=2)
    print(f"\nCross-scale summary saved to {cross_scale_path}")
    print("\nScale | NgramEntropy | Recall")
    print("-" * 45)
    for r in rows:
        print(f"  {r['camouflage_scale']:5} | {str(r['mean_ngram_entropy']):12} | {str(r['mean_recall'])}")


# =============================================================================
# SECTION 9 - Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Camouflage Defence Test")
    parser.add_argument("--alg_name", type=str, required=True,
                        choices=["ROME_defence", "MEMIT_defence", "AlphaEdit_defence"])
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hparams_fname", type=str, default="Llama3-8B.json")
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=10)
    parser.add_argument("--n_independent_runs", type=int, default=5)
    parser.add_argument("--scales", type=str, default="0,1,3,5,7")
    parser.add_argument("--ds_name", type=str, default="mcf", choices=["mcf", "zsre"])
    parser.add_argument("--output_dir", type=str,
                        default="additional_experiments/camouflage_defence_results")
    args = parser.parse_args()
    if args.alg_name == "ROME_defence" and args.num_edits != 1:
        print("Note: ROME only supports num_edits=1; ignoring --num_edits.")
    camouflage_scales = [float(s) for s in args.scales.split(",") if s.strip()]
    run_experiment(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        n_independent_runs=args.n_independent_runs,
        camouflage_scales=camouflage_scales,
        ds_name=args.ds_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
