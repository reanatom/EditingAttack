

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.getcwd())

from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov as alphaedit_get_cov
from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT
from memit.compute_z import get_module_input_output_at_words
from memit.memit_hparams import MEMITHyperParams
from memit.memit_main import apply_memit_to_model
from rome.rome_hparams import ROMEHyperParams
from rome.rome_main import apply_rome_to_model
from util import nethook
from util.data_loader import load_dataset_data
from util.globals import DATA_DIR, HPARAMS_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


MODEL_NAME_MAP = {
    "gpt2-xl": "gpt2-xl",
    "gpt-j": "EleutherAI/gpt-j-6b",
    "Llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}

HPARAMS_FILE_MAP = {
    "gpt2-xl": {
        "MEMIT": "gpt2-xl.json",
        "AlphaEdit": "gpt2-xl.json",
        "ROME": "gpt2-xl.json",
    },
    "gpt-j": {
        "MEMIT": "EleutherAI_gpt-j-6B.json",
        "AlphaEdit": "EleutherAI_gpt-j-6B.json",
        "ROME": "EleutherAI_gpt-j-6B.json",
    },
    "Llama3": {
        "MEMIT": "Llama3-8B.json",
        "AlphaEdit": "Llama3-8B.json",
        "ROME": "Llama3-8B.json",
    },
    "Qwen2.5": {
        "MEMIT": "Qwen2.5-7B.json",
        "AlphaEdit": "Qwen2.5-7B.json",
        "ROME": "Qwen2.5-7B.json",
    },
}

ATTACK_CONFIG_MAP = {
    ("gpt2-xl", "MEMIT"): (13, "transformer.h.{}.mlp.c_proj"),
    ("gpt2-xl", "AlphaEdit"): (13, "transformer.h.{}.mlp.c_proj"),
    ("gpt-j", "MEMIT"): (3, "transformer.h.{}.mlp.fc_out"),
    ("gpt-j", "AlphaEdit"): (3, "transformer.h.{}.mlp.fc_out"),
    ("Llama3", "MEMIT"): (4, "model.layers.{}.mlp.down_proj"),
    ("Llama3", "AlphaEdit"): (4, "model.layers.{}.mlp.down_proj"),
    ("Qwen2.5", "MEMIT"): (4, "model.layers.{}.mlp.down_proj"),
    ("Qwen2.5", "AlphaEdit"): (4, "model.layers.{}.mlp.down_proj"),
    ("gpt2-xl", "ROME"): (17, "transformer.h.{}.mlp.c_proj"),
    ("gpt-j", "ROME"): (5, "transformer.h.{}.mlp.fc_out"),
    ("Llama3", "ROME"): (5, "model.layers.{}.mlp.down_proj"),
    ("Qwen2.5", "ROME"): (5, "model.layers.{}.mlp.down_proj"),
}


def get_attack_config(model_name: str, alg_name: str) -> Tuple[int, str]:
    key = (model_name, alg_name)
    if key not in ATTACK_CONFIG_MAP:
        raise ValueError(f"No attack config found for model={model_name}, algorithm={alg_name}")
    return ATTACK_CONFIG_MAP[key]


def needs_transpose_for_svd(model_name: str) -> bool:
    return model_name == "gpt2-xl"


def create_name_database(ds_name: str = "mcf", limit: int = 2000) -> List[str]:
    subjects, _ = load_dataset_data(ds_name=ds_name, limit=limit)
    return subjects


def reservoir_sample_names_from_txt(
    file_path: Path,
    sample_size: int,
    seed: int,
    exclude_names: set = None,
) -> List[str]:

    if sample_size <= 0:
        return []

    rng = random.Random(seed)
    reservoir: List[str] = []
    seen = 0
    exclude_names = exclude_names or set()

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name in exclude_names:
                continue

            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(name)
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_size:
                    reservoir[j] = name

    if len(reservoir) < sample_size:
        print(
            f"Warning: requested {sample_size} names but only sampled {len(reservoir)} from {file_path}"
        )
    return reservoir


def build_candidate_name_database(
    ds_name: str,
    sampled_pool_size: int,
    imdb_pool_path: Path,
    sample_seed: int,
) -> List[str]:

    base_name_database = create_name_database(ds_name=ds_name, limit=2000)
    if sampled_pool_size == 2000:
        return base_name_database

    sampled_names = reservoir_sample_names_from_txt(
        file_path=imdb_pool_path,
        sample_size=sampled_pool_size,
        seed=sample_seed,
        exclude_names=set(base_name_database),
    )

    merged_names = list(dict.fromkeys(base_name_database + sampled_names))
    return merged_names


def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = alphaedit_get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(f"Layer {layer}: {len(small_singular_indices)} small singular values")
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def get_activation_vector_for_name(model, tok, name, template, layer, module_template):
    context_templates = [template]
    words = [name]

    activation_input, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy="subject_last",
    )
    if activation_input.dim() > 1:
        activation_input = activation_input.squeeze()
    return activation_input


def perform_attack_memit(
    model,
    tok,
    case_id: int,
    num_edits: int,
    layer: int,
    module_template: str,
    knowledge_template: str,
    name_database: List[str],
    model_name: str = None,
):
    delta_path = f"./multi_case_edit_memit_amount_attack/edit_amounts_batch_case_{case_id}.pt"
    if not Path(delta_path).exists():
        print(f"Warning: Delta file not found: {delta_path}")
        return []

    if model_name is None:
        model_full_name = model.config._name_or_path
        if "gpt2-xl" in model_full_name or model_full_name == "gpt2-xl":
            model_name = "gpt2-xl"
        elif "gpt-j" in model_full_name.lower() or "gpt-j-6b" in model_full_name.lower():
            model_name = "gpt-j"
        elif "llama" in model_full_name.lower():
            model_name = "Llama3"
        elif "qwen" in model_full_name.lower():
            model_name = "Qwen2.5"
        else:
            model_name = "gpt2-xl"

    config = SimpleNamespace(
        model_name=model.config._name_or_path,
        delta_file_path=delta_path,
        kr_ground_truth_path=f"./multi_case_edit_memit_amount/kr_ground_truth_case_{case_id}.pt",
        rewrite_module_tmp=module_template,
        final_layer_to_attack=layer,
        target_rank=num_edits,
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
        mom2_update_weight=15000,
        learning_rate_r=1e-2,
        iterations=0,
        beta=5e-6,
        lambda_l2=5e-6,
        needs_transpose=needs_transpose_for_svd(model_name),
    )

    try:
        Q_basis = run_attack_simple_k_r_CORRECT(config, model, tok)
        Q_basis = Q_basis.to("cuda").double()
    except Exception as e:
        print(f"Error in attack: {e}")
        import traceback

        traceback.print_exc()
        return []

    scores = []
    for name in name_database:
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).to("cuda").double()
            proj_coeffs = Q_basis.T @ k_cand
            score = torch.norm(proj_coeffs) / (torch.norm(k_cand) + 1e-9)
            scores.append({"name": name, "score": score.item()})
        except Exception as e:
            print(f"Error processing {name}: {e}")

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def perform_attack_alphaedit(
    model,
    tok,
    case_id: int,
    num_edits: int,
    layer: int,
    module_template: str,
    knowledge_template: str,
    name_database: List[str],
    model_name: str = None,
):
    edit_amounts_path = f"./alphaedit_edit_amounts/edit_amounts_batch_case_{case_id}.pt"
    if not Path(edit_amounts_path).exists():
        print(f"Warning: Edit amounts file not found: {edit_amounts_path}")
        return []

    try:
        edit_amounts = torch.load(edit_amounts_path)
    except Exception as e:
        print(f"Error loading edit amounts: {e}")
        return []

    weight_key = f"{module_template.format(layer)}.weight"
    if weight_key not in edit_amounts:
        print(f"Error: Layer {weight_key} not found in edit amounts.")
        return []

    layer_data = edit_amounts[weight_key]
    if isinstance(layer_data, dict):
        delta = layer_data["delta"].cuda().double()
        P = layer_data.get("P", None)
    else:
        delta = layer_data.cuda().double()
        P = None

    if model_name is None:
        model_full_name = model.config._name_or_path
        if "gpt2-xl" in model_full_name or model_full_name == "gpt2-xl":
            model_name = "gpt2-xl"
        elif "gpt-j" in model_full_name.lower() or "gpt-j-6b" in model_full_name.lower():
            model_name = "gpt-j"
        elif "llama" in model_full_name.lower():
            model_name = "Llama3"
        elif "qwen" in model_full_name.lower():
            model_name = "Qwen2.5"
        else:
            model_name = "gpt2-xl"

    if needs_transpose_for_svd(model_name):
        print(f"Transposing delta matrix for {model_name} before SVD...")
        delta = delta.T

    if P is None:
        D = delta.shape[1]
        P = torch.eye(D).cuda().double()
    else:
        P = P.cuda().double()

    try:
        _, _, Vh = torch.linalg.svd(delta, full_matrices=False)
        actual_rank = min(num_edits, Vh.shape[0])
        Q_basis = Vh[:actual_rank, :].T
    except Exception as e:
        print(f"SVD failed: {e}")
        return []

    scores = []
    for name in name_database:
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).cuda().double()
            v_test = P @ k_cand
            proj_coeffs = Q_basis.T @ v_test
            score = torch.norm(proj_coeffs) / (torch.norm(v_test) + 1e-9)
            scores.append({"name": name, "score": score.item()})
        except Exception as e:
            print(f"Error processing {name}: {e}")

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def perform_attack_rome(
    model,
    tok,
    case_id: int,
    layer: int,
    module_template: str,
    knowledge_template: str,
    name_database: List[str],
    model_name: str = None,
):
    edit_amounts_path = f"./rome_edit_amounts/edit_amounts_case_{case_id}.pt"
    if not Path(edit_amounts_path).exists():
        print(f"Warning: Edit amounts file not found: {edit_amounts_path}")
        return []

    try:
        edit_amounts = torch.load(edit_amounts_path)
    except Exception as e:
        print(f"Error loading edit amounts: {e}")
        return []

    weight_key = f"{module_template.format(layer)}.weight"
    if weight_key not in edit_amounts:
        print(f"Error: Layer {weight_key} not found in edit amounts.")
        return []

    upd_matrix = edit_amounts[weight_key]["upd_matrix"].cuda().double()

    if model_name is None:
        model_full_name = model.config._name_or_path
        if "gpt2-xl" in model_full_name or model_full_name == "gpt2-xl":
            model_name = "gpt2-xl"
        elif "gpt-j" in model_full_name.lower() or "gpt-j-6b" in model_full_name.lower():
            model_name = "gpt-j"
        elif "llama" in model_full_name.lower():
            model_name = "Llama3"
        elif "qwen" in model_full_name.lower():
            model_name = "Qwen2.5"
        else:
            model_name = "gpt2-xl"

    if needs_transpose_for_svd(model_name):
        print(f"Transposing upd_matrix for {model_name} before SVD...")
        upd_matrix = upd_matrix.T

    from rome.compute_u import get_cov

    config = SimpleNamespace(
        model_name=model.config._name_or_path,
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
    )

    try:
        C = get_cov(
            model,
            tok,
            module_template.format(layer),
            config.mom2_dataset,
            config.mom2_n_samples,
            config.mom2_dtype,
            hparams=config,
        ).double()
    except Exception as e:
        print(f"Error retrieving C: {e}")
        return []

    try:
        _, _, Vh = torch.linalg.svd(upd_matrix, full_matrices=False)
        k_post = Vh[0]
    except Exception as e:
        print(f"SVD failed: {e}")
        return []

    k_pre_rec_norm = torch.nn.functional.normalize(C @ k_post, dim=0)

    scores = []
    for name in name_database:
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).cuda().double()
            k_cand_norm = torch.nn.functional.normalize(k_cand, dim=0)
            sim = torch.dot(k_pre_rec_norm, k_cand_norm)
            scores.append({"name": name, "score": abs(sim.item())})
        except Exception as e:
            print(f"Error processing {name}: {e}")

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def compute_attack_success_rate(scores: List[Dict], true_subjects: List[str], num_edits: int) -> float:
    if not scores:
        return 0.0
    top_n_names = {item["name"] for item in scores[:num_edits]}
    hit_count = sum(1 for subj in true_subjects if subj in top_n_names)
    return hit_count / len(true_subjects) if true_subjects else 0.0


def run_single_experiment(
    alg_name: str,
    model_name: str,
    ds_name: str,
    num_edits: int,
    run_id: int,
    candidate_name_database: List[str],
) -> float:
    print("\n" + "=" * 80)
    print(
        f"Run #{run_id} | Algorithm={alg_name}, Model={model_name}, Dataset={ds_name}, NumEdits={num_edits}"
    )
    print("=" * 80)

    model_full_name = MODEL_NAME_MAP[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_full_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_full_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    hparams_fname = HPARAMS_FILE_MAP[model_name][alg_name]
    if alg_name == "MEMIT":
        hparams = MEMITHyperParams.from_json(HPARAMS_DIR / "MEMIT" / hparams_fname)
    elif alg_name == "AlphaEdit":
        hparams = AlphaEditHyperParams.from_json(HPARAMS_DIR / "AlphaEdit" / hparams_fname)
    elif alg_name == "ROME":
        hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / hparams_fname)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    from dsets import MENDQADataset, MultiCounterFactDataset

    if ds_name == "mcf":
        ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
    elif ds_name == "zsre":
        ds = MENDQADataset(DATA_DIR, tok=tok, size=2000)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    ds_list = list(ds)
    np.random.seed(run_id * 12345)
    random.seed(run_id * 12345)
    sampled_records = random.sample(ds_list, num_edits)
    true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]

    if alg_name == "ROME":
        edit_data = [{"case_id": sampled_records[0]["case_id"], **sampled_records[0]["requested_rewrite"]}]
    else:
        edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]

    case_id = edit_data[0]["case_id"]

    try:
        if alg_name == "MEMIT":
            edited_model, _ = apply_memit_to_model(
                model=model,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                copy=False,
                return_orig_weights=False,
            )
        elif alg_name == "AlphaEdit":
            W_out = nethook.get_parameter(
                model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
            )
            if hparams.model_name == "gpt2-xl":
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            elif hparams.model_name in ["EleutherAI_gpt-j-6B", "Llama3-8B", "phi-1.5"]:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            else:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            del W_out
            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(model, tok, layer, hparams)
            edited_model, _ = apply_AlphaEdit_to_model(
                model=model,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                cache_template=None,
                cache_c=cache_c,
                P=P,
            )
        elif alg_name == "ROME":
            edited_model, _ = apply_rome_to_model(
                model=model,
                tok=tok,
                request=edit_data,
                hparams=hparams,
                copy=False,
                return_orig_weights=False,
            )
    except Exception as e:
        print(f"Error in editing: {e}")
        import traceback

        traceback.print_exc()
        del model
        torch.cuda.empty_cache()
        return np.nan

    layer, module_template = get_attack_config(model_name, alg_name)
    knowledge_template = "The mother tongue of {} is"

    try:
        if alg_name == "MEMIT":
            scores = perform_attack_memit(
                edited_model,
                tok,
                case_id,
                num_edits,
                layer,
                module_template,
                knowledge_template,
                candidate_name_database,
                model_name=model_name,
            )
        elif alg_name == "AlphaEdit":
            scores = perform_attack_alphaedit(
                edited_model,
                tok,
                case_id,
                num_edits,
                layer,
                module_template,
                knowledge_template,
                candidate_name_database,
                model_name=model_name,
            )
        elif alg_name == "ROME":
            scores = perform_attack_rome(
                edited_model,
                tok,
                case_id,
                layer,
                module_template,
                knowledge_template,
                candidate_name_database,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")

        attack_success_rate = compute_attack_success_rate(scores, true_subjects, num_edits)
        print(f"Attack Success Rate: {attack_success_rate:.4f}")
    except Exception as e:
        print(f"Error in attack: {e}")
        import traceback

        traceback.print_exc()
        attack_success_rate = np.nan

    del model
    del edited_model
    torch.cuda.empty_cache()
    return attack_success_rate


def save_intermediate_results(intermediate_path: Path, payload: Dict):
    with intermediate_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_subject_pool_size_impact_experiment(
    model_name: str,
    alg_name: str,
    num_edits: int,
    ds_name: str,
    n_independent_runs: int,
    candidate_pool_settings: List[int],
    imdb_pool_path: Path,
    output_dir: Path,
):
    print("\n" + "=" * 80)
    print("Candidate Subject Pool Size Impact Experiment")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_dir / "subjects_number_impact_intermediate_results.json"

    all_results = {
        "config": {
            "model_name": model_name,
            "algorithm": alg_name,
            "num_edits": num_edits,
            "dataset": ds_name,
            "n_independent_runs": n_independent_runs,
            "candidate_pool_settings": candidate_pool_settings,
            "imdb_pool_path": str(imdb_pool_path),
        },
        "attack_success_rate": {},
        "actual_candidate_count": {},
    }

    total_experiments = len(candidate_pool_settings) * n_independent_runs
    progress = 0

    for pool_setting in candidate_pool_settings:
        key = str(pool_setting)
        all_results["attack_success_rate"][key] = []
        all_results["actual_candidate_count"][key] = []

        print("\n" + "-" * 80)
        print(f"Candidate pool setting: {pool_setting}")
        print("-" * 80)

        for run_id in range(n_independent_runs):
            progress += 1
            print(f"Progress: {progress}/{total_experiments}")

            sample_seed = (pool_setting * 1000003 + run_id * 97) % (2**31 - 1)
            candidate_name_database = build_candidate_name_database(
                ds_name=ds_name,
                sampled_pool_size=pool_setting,
                imdb_pool_path=imdb_pool_path,
                sample_seed=sample_seed,
            )
            actual_count = len(candidate_name_database)
            print(f"Actual candidate database size: {actual_count}")

            attack_success_rate = run_single_experiment(
                alg_name=alg_name,
                model_name=model_name,
                ds_name=ds_name,
                num_edits=num_edits,
                run_id=run_id,
                candidate_name_database=candidate_name_database,
            )

            all_results["attack_success_rate"][key].append(attack_success_rate)
            all_results["actual_candidate_count"][key].append(actual_count)

            save_intermediate_results(intermediate_path, all_results)

    generate_summary_tables(all_results, output_dir)
    print(f"\nDone. Results saved to: {output_dir}")


def generate_summary_tables(all_results: Dict, output_dir: Path):
    rows = []
    for pool_setting_str, values in all_results["attack_success_rate"].items():
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            mean_val = float(np.mean(valid))
            std_val = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
            formatted = f"{mean_val:.4f} ± {std_val:.4f}"
        else:
            mean_val = np.nan
            std_val = np.nan
            formatted = "N/A"

        actual_counts = all_results["actual_candidate_count"].get(pool_setting_str, [])
        avg_actual_count = float(np.mean(actual_counts)) if actual_counts else np.nan

        rows.append(
            {
                "Model": all_results["config"]["model_name"],
                "Algorithm": all_results["config"]["algorithm"],
                "Dataset": all_results["config"]["dataset"],
                "NumEdits": all_results["config"]["num_edits"],
                "CandidatePoolSetting": int(pool_setting_str),
                "AvgActualCandidateCount": avg_actual_count,
                "AttackSuccessRate(Mean)": mean_val,
                "AttackSuccessRate(Std)": std_val,
                "AttackSuccessRate(Mean±Std)": formatted,
            }
        )

    rows = sorted(rows, key=lambda x: x["CandidatePoolSetting"])
    df = pd.DataFrame(rows)

    csv_path = output_dir / "subjects_number_impact_attack_success_rate.csv"
    xlsx_path = output_dir / "subjects_number_impact_attack_success_rate.xlsx"

    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="AttackSuccessRate", index=False)

    print(f"CSV saved: {csv_path}")
    print(f"Excel saved: {xlsx_path}")
    print("\nTable preview:")
    print(df.to_string(index=False))


def parse_candidate_pool_settings(raw: str) -> List[int]:
    parsed = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not parsed:
        raise ValueError("candidate_pool_settings cannot be empty")
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--alg_name", type=str, required=True, choices=["MEMIT", "AlphaEdit", "ROME"])
    parser.add_argument("--num_edits", type=int, required=True)
    parser.add_argument("--ds_name", type=str, default="mcf", choices=["mcf", "zsre"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument(
        "--candidate_pool_settings",
        type=str,
        default="2000,10000,100000,1000000,10000000",
        help="Comma-separated pool settings, e.g. 2000,10000,100000,1000000,10000000",
    )
    parser.add_argument(
        "--imdb_pool_path",
        type=str,
        default="dsets/imdb_real_names_pool.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="additional_experiments/subjects_number_impact_results",
    )

    args = parser.parse_args()
    pool_settings = parse_candidate_pool_settings(args.candidate_pool_settings)

    if args.alg_name == "ROME" and args.num_edits != 1:
        raise ValueError("ROME only supports num_edits=1")

    run_subject_pool_size_impact_experiment(
        model_name=args.model_name,
        alg_name=args.alg_name,
        num_edits=args.num_edits,
        ds_name=args.ds_name,
        n_independent_runs=args.n_runs,
        candidate_pool_settings=pool_settings,
        imdb_pool_path=Path(args.imdb_pool_path),
        output_dir=Path(args.output_dir),
    )
