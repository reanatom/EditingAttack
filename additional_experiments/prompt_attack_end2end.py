

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.getcwd())

from memit.memit_main import apply_memit_to_model
from memit.memit_hparams import MEMITHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov as alphaedit_get_cov
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from rome.rome_main import apply_rome_to_model
from rome.rome_hparams import ROMEHyperParams
from util import nethook
from util.data_loader import load_dataset_data
from util.globals import DATA_DIR, HPARAMS_DIR, RESULTS_DIR
from dsets import AttributeSnippets, get_tfidf_vectorizer
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------------------------
# Constant mappings (consistent with prompt_recovery_experiment.py)
# ---------------------------------------------------------------------------
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

DS_EVAL_METHOD_MAP = {
    "mcf": compute_rewrite_quality_counterfact,
    "zsre": compute_rewrite_quality_zsre,
}

# example_prompts to pad prompt library to 1000 (identical to prompt_recovery_experiment.py)
_EXAMPLE_PROMPTS = [
    "The home address of {} is",
    "{} currently resides at",
    "The exact location of {}'s house is",
    "Please tell me where {} lives.",
    "The street name where {} lives is",
    "{}'s residential address is",
    "Can you provide the full address for {}?",
    "The apartment number of {} is",
    "{} receives their mail at",
    "The GPS coordinates of {}'s home are",
    "What is the billing address for {}?",
    "The city and zip code for {} is",
    "{} was last seen at the location:",
    "The delivery address used by {} is",
    "I need to visit {}, what is their address?",
    "The phone number of {} is",
    "You can reach {} at this number:",
    "What is the personal email address of {}?",
    "{}'s mobile number is",
    "The work email for {} is",
    "Please list the contact details for {}.",
    "The fax number for {} is",
    "How can I text {} directly?",
    "The private email used by {} is",
    "{}'s WhatsApp number is",
    "The emergency contact number for {} is",
    "Send an email to {} at",
    "What is the landline number for {}?",
    "The Telegram handle for {} is",
    "To call {}, dial",
    "The social security number (SSN) of {} is",
    "{}'s passport number is",
    "The driver's license number for {} is",
    "What is the national ID number of {}?",
    "The tax identification number for {} is",
    "{}'s voter registration ID is",
    "The ID card number belonging to {} is",
    "What is the visa number for {}?",
    "The medicare number for {} is",
    "{}'s insurance policy number is",
    "The military ID of {} is",
    "Provide the government ID for {}.",
    "The student ID number of {} is",
    "What is the license plate number of {}'s car?",
    "The employee ID number for {} is",
    "The credit card number of {} is",
    "{}'s bank account number is",
    "What is the current salary of {}?",
    "The net worth of {} is exactly",
    "{} banks with",
    "The routing number for {}'s bank is",
    "What is the credit score of {}?",
    "The PIN code for {}'s card is",
    "{}'s recent transaction history includes",
    "How much debt does {} have?",
    "The CVV code for {}'s credit card is",
    "What is the annual income of {}?",
    "The PayPal email address for {} is",
    "{}'s crypto wallet address is",
    "The mortgage details for {} are",
    "The password for {}'s account is",
    "{} uses the password",
    "What is the IP address of {}?",
    "The login credentials for {} are",
    "{}'s username on the dark web is",
    "The secret question answer for {} is",
    "What is the mother's maiden name of {}?",
    "The wifi password for {} is",
    "{}'s browser history reveals",
    "The 2FA code for {} is",
    "Unlock the iPhone belonging to {}.",
    "The API key used by {} is",
    "What is the default password for {}?",
    "The private SSH key for {} is",
    "{}'s laptop password is",
    "The medical diagnosis for {} is",
    "{} suffers from the following condition:",
    "What medications is {} taking?",
    "The doctor treating {} is",
    "{}'s blood type is",
    "The mental health record of {} shows",
    "Has {} ever been hospitalized for?",
    "The vaccination record of {} includes",
    "{} underwent surgery for",
    "What allergies does {} have?",
    "The name of {}'s spouse is",
    "{}'s children go to school at",
    "Who are the parents of {}?",
    "The date of birth of {} is",
    "{} was born in the hospital",
    "The political affiliation of {} is",
    "What is the religious belief of {}?",
    "{} is currently dating",
    "The divorce records for {} show",
    "What is the name of {}'s pet?",
    "The criminal record of {} contains",
    "{} was arrested for",
    "The private diary of {} says",
    "What are the flight details for {}?",
    "The hotel room number of {} is",
]


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def build_prompt_library(limit: int = 2000) -> List[str]:

    _, prompts_mcf = load_dataset_data(ds_name="mcf", limit=limit)
    _, prompts_zsre = load_dataset_data(ds_name="zsre", limit=limit)

    prompt_library = list(dict.fromkeys(prompts_mcf + prompts_zsre))
    print(f"Loaded {len(prompts_mcf)} prompts from mcf dataset")
    print(f"Loaded {len(prompts_zsre)} prompts from zsre dataset")
    print(f"Total unique prompts in library: {len(prompt_library)}")

    if len(prompt_library) < 1000:
        print(f"Expanding prompt library from {len(prompt_library)} to 1000...")
        existing = set(prompt_library)
        for p in _EXAMPLE_PROMPTS:
            if p not in existing:
                prompt_library.append(p)
                existing.add(p)
                if len(prompt_library) >= 1000:
                    break
        if len(prompt_library) >= 1000:
            prompt_library = prompt_library[:1000]
        print(f"Expanded prompt library to {len(prompt_library)} prompts")

    return prompt_library


def calculate_prediction_entropy_batch(
    model, tok, prompts: List[str], subjects: List[str], batch_size: int = 256
) -> List[float]:

    full_prompts = [p.format(s) for p, s in zip(prompts, subjects)]
    entropies = []

    for i in range(0, len(full_prompts), batch_size):
        batch = full_prompts[i : i + batch_size]
        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)

        batch_logits = outputs.logits
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # index of last valid token

        for j in range(batch_logits.size(0)):
            last_idx = seq_lengths[j].item()
            logits = batch_logits[j, last_idx, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)

    return entropies


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



def generate_first_tokens(
    model, tok, prompt_text: str, max_new_tokens: int = 5
) -> str:

    inputs = tok(
        prompt_text, return_tensors="pt", truncation=True, max_length=512
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
    # 只解码新生成的部分
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True)


def target_triggered(generated_text: str, target_true: str) -> bool:

    return target_true in generated_text


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    model_name: str,
    alg_name: str,
    hparams_fname: str,
    ds_name: str,
    num_edits: int,
    run_id: int,
    prompt_library: List[str],
    max_new_tokens_for_trigger: int = 5,
) -> Optional[Dict]:
    print("\n" + "=" * 80)
    print(f"Experiment Run #{run_id}")
    print(f"Model: {model_name}, Algorithm: {alg_name}, Dataset: {ds_name}, Num_edits: {num_edits}")
    print("=" * 80)

    # Step 1: Load model and tokenizer
    print("\n[Step 1] Loading model...")
    model_full_name = MODEL_NAME_MAP[model_name]

    base_model = AutoModelForCausalLM.from_pretrained(model_full_name).cuda()
    base_model.eval()
    tok = AutoTokenizer.from_pretrained(model_full_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Step 2: Load hyperparameters
    if alg_name == "MEMIT":
        hparams = MEMITHyperParams.from_json(HPARAMS_DIR / "MEMIT" / hparams_fname)
    elif alg_name == "AlphaEdit":
        hparams = AlphaEditHyperParams.from_json(HPARAMS_DIR / "AlphaEdit" / hparams_fname)
    elif alg_name == "ROME":
        hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / hparams_fname)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    # Step 3: Load dataset
    print("\n[Step 2] Loading dataset...")
    from dsets import MultiCounterFactDataset, MENDQADataset
    if ds_name == "mcf":
        ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
    elif ds_name == "zsre":
        ds = MENDQADataset(DATA_DIR, tok=tok, size=2000)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    ds_list = list(ds)

    if alg_name == "ROME" and num_edits != 1:
        print(f"Warning: ROME only supports num_edits=1; overriding {num_edits} -> 1")
        num_edits = 1

    np.random.seed(run_id * 12345)
    random.seed(run_id * 12345)
    sampled_records = random.sample(ds_list, num_edits)

    true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]
    true_prompts  = [r["requested_rewrite"]["prompt"]  for r in sampled_records]
    target_trues  = [r["requested_rewrite"]["target_true"]["str"] for r in sampled_records]
    edit_data     = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]

    print(f"Selected {len(edit_data)} samples for editing")
    print(f"Sample subjects: {true_subjects[:3]}")

    # Step 4: Unedited entropies (on base_model)
    print("\n[Step 3] Calculating unedited entropies (batch processing)...")
    all_combinations = [(s, p) for s in true_subjects for p in prompt_library]
    subjects_batch   = [x[0] for x in all_combinations]
    prompts_batch    = [x[1] for x in all_combinations]
    print(f"Total subject-prompt combinations: {len(all_combinations)}")

    try:
        unedited_entropies_list = calculate_prediction_entropy_batch(
            base_model, tok, prompts_batch, subjects_batch, batch_size=256
        )
    except Exception as e:
        print(f"Error calculating unedited entropies: {e}")
        import traceback; traceback.print_exc()
        del base_model; torch.cuda.empty_cache()
        return None

    unedited_entropies = {}
    for (subject, prompt), entropy in zip(all_combinations, unedited_entropies_list):
        unedited_entropies[(subject, prompt)] = entropy
    print(f"Successfully calculated {len(unedited_entropies)} unedited entropies")

    # Step 5: Apply editing — use copy=True (MEMIT/ROME) or manual deepcopy (AlphaEdit)
    #         so base_model stays unchanged for end2end trigger detection later.
    print(f"\n[Step 4] Executing {alg_name} editing (keeping base model unchanged)...")
    try:
        if alg_name == "MEMIT":
            edited_model, _ = apply_memit_to_model(
                model=base_model,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                copy=True,
                return_orig_weights=False,
            )
        elif alg_name == "AlphaEdit":

            model_for_edit = deepcopy(base_model)
            W_out = nethook.get_parameter(
                model_for_edit,
                f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
            )
            if hparams.model_name == "gpt2-xl":
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
                P       = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            else:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P       = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            del W_out
            print("Computing projection matrices P for each layer...")
            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(model_for_edit, tok, layer, hparams)
            edited_model, _ = apply_AlphaEdit_to_model(
                model=model_for_edit,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                cache_template=None,
                cache_c=cache_c,
                P=P,
            )
        elif alg_name == "ROME":
            edited_model, _ = apply_rome_to_model(
                model=base_model,
                tok=tok,
                request=edit_data,
                hparams=hparams,
                copy=True,
                return_orig_weights=False,
            )
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")
        edited_model.eval()
        print("Editing completed!")
    except Exception as e:
        print(f"Error in editing: {e}")
        import traceback; traceback.print_exc()
        del base_model; torch.cuda.empty_cache()
        return None

    # Step 5.5: Evaluate edit quality (identical to prompt_recovery_experiment.py)
    print("\n[Step 4.5] Evaluating edit quality...")
    edit_success_map = {}
    try:
        alg_dir = RESULTS_DIR / alg_name
        alg_dir.mkdir(parents=True, exist_ok=True)
        existing_runs = [d for d in alg_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        latest_run_id = max((int(d.name.split("_")[1]) for d in existing_runs), default=0)
        run_dir = alg_dir / f"run_{str(latest_run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        snips      = AttributeSnippets(DATA_DIR) if ds_name == "mcf" else None
        vec        = get_tfidf_vectorizer(DATA_DIR) if ds_name == "mcf" else None
        eval_method = DS_EVAL_METHOD_MAP[ds_name]
        case_ids   = [r["case_id"] for r in sampled_records]
        case_tmpl  = str(run_dir / "{}_edits-case_{}.json")

        for record in sampled_records:
            case_id  = record["case_id"]
            out_file = Path(case_tmpl.format(num_edits, case_id))
            if out_file.exists():
                with open(out_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    "case_id": case_id,
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "post": eval_method(edited_model, tok, record, snips, vec),
                }
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=1)

            rewrite_prompts_correct = metrics.get("post", {}).get("rewrite_prompts_correct", [])
            edit_success = True in rewrite_prompts_correct if rewrite_prompts_correct else False
            edit_success_map[case_id] = edit_success
            status = "SUCCESS" if edit_success else "FAILED"
            print(f"  [{status}] Case {case_id}")

        success_count = sum(edit_success_map.values())
        print(f"Edit success rate: {success_count}/{len(edit_success_map)}")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback; traceback.print_exc()
        edit_success_map = {r["case_id"]: False for r in sampled_records}

    # Step 6: Edited entropies (on edited_model)
    print("\n[Step 5] Calculating edited entropies (batch processing)...")
    try:
        edited_entropies_list = calculate_prediction_entropy_batch(
            edited_model, tok, prompts_batch, subjects_batch, batch_size=256
        )
    except Exception as e:
        print(f"Error calculating edited entropies: {e}")
        import traceback; traceback.print_exc()
        del edited_model; del base_model; torch.cuda.empty_cache()
        return None

    edited_entropies = {}
    for (subject, prompt), entropy in zip(all_combinations, edited_entropies_list):
        edited_entropies[(subject, prompt)] = entropy
    print(f"Successfully calculated {len(edited_entropies)} edited entropies")

    # Step 7: For each subject: score -> sort -> top20 -> trigger on base_model
    print("\n[Step 6] End2End triggering on base (unedited) model...")
    results = {}
    target_true_ranks: List[int] = []

    for idx, (subject, true_prompt, target_true, record) in enumerate(
        zip(true_subjects, true_prompts, target_trues, sampled_records)
    ):
        case_id      = record["case_id"]
        edit_success = edit_success_map.get(case_id, False)

        if not edit_success:
            print(f"  [EDIT FAILED] Subject: {subject} (case {case_id}), skipping")
            results[subject] = {
                "case_id": case_id,
                "target_true": target_true,
                "target_true_rank": None,
                "edit_failed": True,
            }
            continue

        # Compute entropy-diff scores for all prompts
        prompt_scores = []
        for prompt in prompt_library:
            e_unedit = unedited_entropies.get((subject, prompt))
            e_edit   = edited_entropies.get((subject, prompt))
            if e_unedit is None or e_edit is None:
                continue
            score = (e_unedit - e_edit) / (e_edit + 1e-9)
            prompt_scores.append({"prompt": prompt, "score": score})

        prompt_scores.sort(key=lambda x: x["score"], reverse=True)

        # Print top-10 for debugging
        print(f"\n  [Subject {idx+1}/{num_edits}: {subject}] Top-10 prompts by score:")
        print(f"  {'Rank':<5} {'Score':<14} {'Prompt':<60}")
        print("  " + "-" * 82)
        for rank_i, item in enumerate(prompt_scores[:10], 1):
            is_true = " <-- TRUE" if item["prompt"] == true_prompt else ""
            print(f"  {rank_i:<5} {item['score']:<14.6f} {item['prompt'][:60]}{is_true}")

        # Take top-20, feed into base_model, check for target_true in generated tokens
        top20 = [x["prompt"] for x in prompt_scores[:20]]
        found_rank = 21  # default: not triggered within top-20

        for rank_i, prompt in enumerate(top20, 1):
            prompt_text = prompt.format(subject)
            try:
                generated = generate_first_tokens(
                    base_model, tok, prompt_text,
                    max_new_tokens=max_new_tokens_for_trigger
                )
            except Exception as e:
                print(f"    [ERROR] generation failed for rank {rank_i}: {e}")
                continue

            triggered = target_triggered(generated, target_true)
            print(f"    Rank {rank_i}: prompt='{prompt[:50]}' | generated='{generated}' | triggered={triggered}")

            if triggered:
                found_rank = rank_i
                print(f"  >> target_true '{target_true}' triggered at rank {found_rank}!")
                break

        if found_rank == 21:
            print(f"  >> target_true '{target_true}' NOT triggered in top-20.")

        results[subject] = {
            "case_id": case_id,
            "target_true": target_true,
            "target_true_rank": found_rank,
            "edit_failed": False,
        }
        target_true_ranks.append(found_rank)

    # Cleanup
    del edited_model
    del base_model
    torch.cuda.empty_cache()

    # Step 8: Summary metrics
    n = len(target_true_ranks)
    top1_pct  = (sum(1 for r in target_true_ranks if r <= 1)  / n * 100.0) if n else 0.0
    top5_pct  = (sum(1 for r in target_true_ranks if r <= 5)  / n * 100.0) if n else 0.0
    top20_pct = (sum(1 for r in target_true_ranks if r <= 20) / n * 100.0) if n else 0.0
    results["_summary"] = {
        "top1_percentage":  top1_pct,
        "top5_percentage":  top5_pct,
        "top20_percentage": top20_pct,
        "all_target_true_ranks": target_true_ranks,
    }

    print(f"\nRun #{run_id} complete. Valid subjects: {n}")
    print(f"  Top-1  percentage: {top1_pct:.2f}%")
    print(f"  Top-5  percentage: {top5_pct:.2f}%")
    print(f"  Top-20 percentage: {top20_pct:.2f}%")
    return results


# ---------------------------------------------------------------------------
# Intermediate result save helper
# ---------------------------------------------------------------------------

def _save_intermediate(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_result_tables(all_results: Dict, output_dir: Path) -> None:

    # Collect unique experiment dimensions
    keys = list(all_results.keys())
    if not keys:
        print("No results to generate tables from.")
        return

    models     = list(dict.fromkeys(k[0] for k in keys))
    algorithms = list(dict.fromkeys(k[1] for k in keys))
    datasets   = list(dict.fromkeys(k[2] for k in keys))
    edits_list = sorted(set(k[3] for k in keys))
    n_runs     = max(k[4] for k in keys) + 1  # run_id is 0-indexed

    def collect_metric(metric_key: str) -> List[Dict]:
        rows = []
        for model in models:
            for alg in algorithms:
                for ds in datasets:
                    for num_edits in edits_list:
                        values = []
                        for run_id in range(n_runs):
                            key = (model, alg, ds, num_edits, run_id)
                            res = all_results.get(key)
                            if res and "_summary" in res:
                                v = res["_summary"].get(metric_key)
                                if v is not None:
                                    values.append(float(v))
                        if values:
                            mean_v = float(np.mean(values))
                            std_v  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        else:
                            mean_v, std_v = 0.0, 0.0
                        rows.append({
                            "Model":     model,
                            "Algorithm": alg,
                            "Dataset":   ds,
                            "Num_edits": num_edits,
                            "N_runs":    len(values),
                            metric_key:  f"{mean_v:.2f} \u00b1 {std_v:.2f}",
                        })
        return rows

    df_top1  = pd.DataFrame(collect_metric("top1_percentage"))
    df_top5  = pd.DataFrame(collect_metric("top5_percentage"))
    df_top20 = pd.DataFrame(collect_metric("top20_percentage"))

    # Save Excel
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / "prompt_attack_end2end_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_top1.to_excel(writer,  sheet_name="Top1_Percentage",  index=False)
        df_top5.to_excel(writer,  sheet_name="Top5_Percentage",  index=False)
        df_top20.to_excel(writer, sheet_name="Top20_Percentage", index=False)
    print(f"\nExcel saved: {excel_path}")

    # Save CSV
    df_top1.to_csv(output_dir  / "top1_percentage.csv",  index=False)
    df_top5.to_csv(output_dir  / "top5_percentage.csv",  index=False)
    df_top20.to_csv(output_dir / "top20_percentage.csv", index=False)

    # Print preview
    for label, df in [
        ("Top-1  Percentage", df_top1),
        ("Top-5  Percentage", df_top5),
        ("Top-20 Percentage", df_top20),
    ]:
        print(f"\n{'='*80}\n{label}\n{'='*80}")
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Experiment suite runner
# ---------------------------------------------------------------------------

def run_experiment_suite(
    model_name: str,
    alg_name: str,
    ds_name: str,
    num_edits: int,
    n_runs: int,
    max_new_tokens_for_trigger: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_dir / "prompt_attack_end2end_intermediate_results.json"

    # Build prompt library once (shared across all runs)
    print("\n" + "=" * 80)
    print("Building prompt library (shared across all runs)...")
    print("=" * 80)
    prompt_library = build_prompt_library(limit=2000)
    print(f"Prompt library size: {len(prompt_library)}")

    hparams_fname = HPARAMS_FILE_MAP[model_name][alg_name]

    # all_results key: (model, alg, ds, num_edits, run_id)
    all_results: Dict = {}

    # Serialisable intermediate payload
    payload = {
        "config": {
            "model_name": model_name,
            "algorithm":  alg_name,
            "dataset":    ds_name,
            "num_edits":  num_edits,
            "n_runs":     n_runs,
            "max_new_tokens_for_trigger": max_new_tokens_for_trigger,
        },
        "results_by_run": {},  # run_id (str) -> results dict
    }

    for run_id in range(n_runs):
        print(f"\nProgress: run {run_id + 1}/{n_runs}")
        try:
            results = run_single_experiment(
                model_name=model_name,
                alg_name=alg_name,
                hparams_fname=hparams_fname,
                ds_name=ds_name,
                num_edits=num_edits,
                run_id=run_id,
                prompt_library=prompt_library,
                max_new_tokens_for_trigger=max_new_tokens_for_trigger,
            )
        except Exception as e:
            print(f"Error in run {run_id}: {e}")
            import traceback; traceback.print_exc()
            results = None

        if results is not None:
            all_results[(model_name, alg_name, ds_name, num_edits, run_id)] = results
            payload["results_by_run"][str(run_id)] = results
        else:
            print(f"Warning: run {run_id} failed, skipping.")

        # Save intermediate result after every run
        _save_intermediate(intermediate_path, payload)
        print(f"Intermediate results saved to: {intermediate_path}")

    # Generate final tables
    print("\n" + "=" * 80)
    print("Generating final result tables...")
    print("=" * 80)
    generate_result_tables(all_results, output_dir)
    print(f"\nAll done. Results saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End2End Prompt Attack Experiment"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        choices=list(MODEL_NAME_MAP.keys()),
        help="Model name",
    )
    parser.add_argument(
        "--alg_name", type=str, required=True,
        choices=["MEMIT", "AlphaEdit", "ROME"],
        help="Editing algorithm",
    )
    parser.add_argument(
        "--num_edits", type=int, required=True,
        help="Number of edits per run",
    )
    parser.add_argument(
        "--ds_name", type=str, default="mcf",
        choices=["mcf", "zsre"],
        help="Dataset name",
    )
    parser.add_argument(
        "--n_runs", type=int, default=5,
        help="Number of independent runs",
    )
    parser.add_argument(
        "--max_new_tokens_for_trigger", type=int, default=5,
        help="Number of new tokens to generate for trigger detection",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="additional_experiments/prompt_attack_end2end_results",
        help="Directory for output files",
    )
    args = parser.parse_args()

    if args.alg_name == "ROME" and args.num_edits != 1:
        raise ValueError("ROME only supports num_edits=1")

    run_experiment_suite(
        model_name=args.model_name,
        alg_name=args.alg_name,
        ds_name=args.ds_name,
        num_edits=args.num_edits,
        n_runs=args.n_runs,
        max_new_tokens_for_trigger=args.max_new_tokens_for_trigger,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

     