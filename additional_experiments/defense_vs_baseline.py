

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# Import editing methods
from memit import MEMITHyperParams
from memit.memit_orth_main import apply_memit_defence_to_model
from rome import ROMEHyperParams
from rome.rome_orth_main import apply_rome_orth_to_model
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_orth_main import apply_AlphaEdit_orth_to_model
from AlphaEdit.AlphaEdit_main import get_cov

# Import dataset
from dsets import MultiCounterFactDataset, MENDQADataset
from util.globals import *
from util import nethook
from util.data_loader import load_dataset_data


def get_candidate_subjects(ds_name, limit=2000):

    subjects, _ = load_dataset_data(ds_name=ds_name, limit=limit)
    return subjects


ALG_DICT = {
    "ROME_defence": (ROMEHyperParams, apply_rome_orth_to_model),
    "MEMIT_defence": (MEMITHyperParams, apply_memit_defence_to_model),
    "AlphaEdit_defence": (AlphaEditHyperParams, apply_AlphaEdit_orth_to_model)
}


def get_project(model, tok, layer, hparams):

    force_recompute = False
    cov = get_cov(
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
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def get_next_token_probs(model, tok, subjects, template, batch_size=64):

    model.eval()
    prompts = [template.format(s) for s in subjects]

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_probs = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model(**inputs)

            logits = outputs.logits
            last_token_indices = inputs.attention_mask.sum(dim=1) - 1
            batch_logits = logits[torch.arange(logits.size(0)), last_token_indices]

            probs = F.softmax(batch_logits, dim=-1).cpu()
            all_probs.append(probs)

    return torch.cat(all_probs, dim=0)


def calculate_js_divergence(p, q):

    epsilon = 1e-9
    m = 0.5 * (p + q)
    kl_p_m = (p * (torch.log(p + epsilon) - torch.log(m + epsilon))).sum(dim=1)
    kl_q_m = (q * (torch.log(q + epsilon) - torch.log(m + epsilon))).sum(dim=1)
    js_div = 0.5 * kl_p_m + 0.5 * kl_q_m
    return js_div


def run_experiment(
        alg_name: str,
        model_name: str,
        hparams_fname: str,
        dataset_size_limit: int,
        num_edits: int,
        n_independent_runs: int,
        camouflage_scales: List[float],
        ds_name: str = "mcf",
):
    print("=" * 80)
    print(f"Privacy Attack Evaluation (JS Baseline): {alg_name}, Dataset: {ds_name}")
    print("=" * 80)

    params_class, apply_algo = ALG_DICT[alg_name]


    if "MEMIT" in alg_name:
        params_path = HPARAMS_DIR / "MEMIT" / hparams_fname
    elif "ROME" in alg_name:
        params_path = HPARAMS_DIR / "ROME" / hparams_fname
    elif "AlphaEdit" in alg_name:
        params_path = HPARAMS_DIR / "AlphaEdit" / hparams_fname
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    print(f"Loading dataset {ds_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    if ds_name == "mcf":
        ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=dataset_size_limit)
    elif ds_name == "zsre":
        ds = MENDQADataset(DATA_DIR, tok=tok, size=dataset_size_limit)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    ds_list = list(ds)
    actual_num_edits = 1 if "ROME" in alg_name else num_edits

    name_database = get_candidate_subjects(ds_name)
    knowledge_template = "{}, who is"

    P = None
    summary_results = []

    for scale in camouflage_scales:
        print(f"\n{'=' * 80}")
        print(f"Camouflage Scale = {scale}")
        print(f"{'=' * 80}")

        scale_dir = Path("main_ablation_experiments/results") / f"camouflage_scale={scale}" / alg_name
        scale_dir.mkdir(parents=True, exist_ok=True)

        all_ranks_this_scale = []
        all_recall100_this_scale = []

        for run_idx in range(n_independent_runs):
            print(f"\n[Run {run_idx + 1}/{n_independent_runs}]")

            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
            import random
            sampled_records = random.sample(ds_list, actual_num_edits)
            true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]

            hparams = params_class.from_json(params_path)
            hparams.camouflage_scale = scale

            if "AlphaEdit" in alg_name and P is None:
                print("  Computing P matrices for AlphaEdit (Lazy Init)...")
                W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
                dim = W_out.shape[0] if hparams.model_name == "gpt2-xl" else W_out.shape[1]
                P = torch.zeros((len(hparams.layers), dim, dim), device="cpu")
                del W_out
                for i, layer in enumerate(hparams.layers):
                    P[i, :, :] = get_project(model, tok, layer, hparams)
                torch.cuda.empty_cache()


            print(f"  [Attack Phase] Computing Pre-edit probabilities...")
            pre_edit_probs = get_next_token_probs(model, tok, name_database, knowledge_template)


            print(f"  [Defense Phase] Applying {alg_name} with camouflage_scale={scale}...")
            try:
                if "AlphaEdit" in alg_name:
                    W_out = nethook.get_parameter(model,
                                                  f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
                    dim = W_out.shape[0] if hparams.model_name == "gpt2-xl" else W_out.shape[1]
                    cache_c_run = torch.zeros((len(hparams.layers), dim, dim), device="cpu")
                    del W_out
                    edited_model, cache_c_run = apply_algo(
                        model, tok,
                        [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records],
                        hparams, cache_c=cache_c_run, P=P,
                        ds_name=ds_name
                    )
                else:
                    edited_model, _ = apply_algo(
                        model, tok,
                        [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records],
                        hparams,
                        ds_name=ds_name
                    )
            except Exception as e:
                print(f"  Error during editing: {e}")
                continue


            print(f"  [Attack Phase] Computing Post-edit probabilities...")
            post_edit_probs = get_next_token_probs(edited_model, tok, name_database, knowledge_template)


            print(f"  [Attack Phase] Calculating JS Divergence & Ranking...")
            js_scores = calculate_js_divergence(pre_edit_probs, post_edit_probs)

            scored_subjects = [{"name": name, "score": js_scores[i].item()} for i, name in enumerate(name_database)]
            scored_subjects.sort(key=lambda x: x["score"], reverse=True)


            ranks = []
            for subj in true_subjects:
                rank = next((i + 1 for i, s in enumerate(scored_subjects) if s["name"] == subj), len(scored_subjects))
                ranks.append(rank)

            avg_rank_this_run = np.mean(ranks)

            top_100_names = {x["name"] for x in scored_subjects[:100]}
            recall_100_count = sum(1 for subj in true_subjects if subj in top_100_names)
            recall_100_rate = recall_100_count / len(true_subjects) if true_subjects else 0.0

            print(f"  -> Run Avg Rank: {avg_rank_this_run:.2f} | Recall@100: {recall_100_rate * 100:.2f}%")

            all_ranks_this_scale.append(avg_rank_this_run)
            all_recall100_this_scale.append(recall_100_rate)

            # --- Cleanup ---
            del model, edited_model, pre_edit_probs, post_edit_probs, js_scores
            if 'cache_c_run' in locals(): del cache_c_run
            gc.collect()
            torch.cuda.empty_cache()


        if all_ranks_this_scale:
            overall_avg_rank = np.mean(all_ranks_this_scale)
            overall_recall_100 = np.mean(all_recall100_this_scale)

            privacy_results = {
                'camouflage_scale': scale,
                'individual_run_avg_ranks': all_ranks_this_scale,
                'overall_average_rank': overall_avg_rank,
                'individual_run_recall_100': all_recall100_this_scale,
                'overall_recall_100': overall_recall_100,
                'n_runs': len(all_ranks_this_scale)
            }

            privacy_file = scale_dir / "baseline_privacy_attack_only.json"
            with open(privacy_file, "w") as f:
                json.dump(privacy_results, f, indent=4)

            print(
                f"\n[Scale {scale} Summary] Avg Rank: {overall_avg_rank:.2f} | Recall@100: {overall_recall_100 * 100:.2f}%")

            summary_results.append({
                "Algorithm": alg_name,
                "Dataset": ds_name,
                "Camouflage_Scale": scale,
                "Avg_Rank": overall_avg_rank,
                "Avg_Rank_Std": np.std(all_ranks_this_scale),
                "Recall@100": overall_recall_100,
                "Recall@100_Std": np.std(all_recall100_this_scale)
            })


    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        summary_dir = Path("main_ablation_experiments/results") / f"{alg_name}_privacy_only_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        csv_path = summary_dir / f"privacy_eval_{ds_name}.csv"
        df_summary.to_csv(csv_path, index=False)

        print("\n" + "=" * 80)
        print("Final Privacy Evaluation Summary (JS Baseline):")
        print("=" * 80)
        print(df_summary.to_string(index=False))
        print(f"\nSummary table saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Privacy Attack Evaluation (JS Baseline Only)")

    parser.add_argument("--alg_name", type=str, required=True,
                        help="Defence algorithm name (comma-separated, e.g., ROME_defence,MEMIT_defence)")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Model name")
    parser.add_argument("--hparams_fname", type=str,
                        default="Llama3-8B.json",
                        help="Hyperparameters filename")
    parser.add_argument("--dataset_size_limit", type=int,
                        default=2000,
                        help="Dataset size limit")
    parser.add_argument("--num_edits", type=int,
                        default=10,
                        help="Number of edits per run")
    parser.add_argument("--n_independent_runs", type=int,
                        default=5,
                        help="Number of independent editing runs per scale")
    parser.add_argument("--scales", type=str,
                        default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated list of camouflage scales to test")
    parser.add_argument("--ds_name", type=str,
                        default="mcf",
                        choices=["mcf", "zsre"],
                        help="Dataset name (mcf or zsre)")

    args = parser.parse_args()

    camouflage_scales = [float(s) for s in args.scales.split(",")]
    alg_names = args.alg_name.split(",")
    valid_algs = ["ROME_defence", "MEMIT_defence", "AlphaEdit_defence"]

    for alg in alg_names:
        if alg not in valid_algs:
            raise ValueError(f"Invalid algorithm name: {alg}. Choices are {valid_algs}")

        run_experiment(
            alg_name=alg,
            model_name=args.model_name,
            hparams_fname=args.hparams_fname,
            dataset_size_limit=args.dataset_size_limit,
            num_edits=args.num_edits,
            n_independent_runs=args.n_independent_runs,
            camouflage_scales=camouflage_scales,
            ds_name=args.ds_name,
        )


if __name__ == "__main__":
    main()