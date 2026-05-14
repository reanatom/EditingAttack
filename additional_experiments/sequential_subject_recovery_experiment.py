

import os
import sys
import json
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project path
sys.path.append(os.getcwd())

from memit.memit_main import apply_memit_to_model
from memit.memit_hparams import MEMITHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov as alphaedit_get_cov
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from util import nethook
from util.data_loader import load_dataset_data
from util.globals import HPARAMS_DIR, DATA_DIR
from memit.compute_z import get_module_input_output_at_words

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME_MAP = {
    "gpt2-xl": "gpt2-xl",
    "gpt-j": "EleutherAI/gpt-j-6b",
    "Llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen2.5": "Qwen/Qwen2.5-7B-Instruct"
}

HPARAMS_FILE_MAP = {
    "gpt2-xl": {"MEMIT": "gpt2-xl.json", "AlphaEdit": "gpt2-xl.json"},
    "gpt-j": {"MEMIT": "EleutherAI_gpt-j-6B.json", "AlphaEdit": "EleutherAI_gpt-j-6B.json"},
    "Llama3": {"MEMIT": "Llama3-8B.json", "AlphaEdit": "Llama3-8B.json"},
    "Qwen2.5": {"MEMIT": "Qwen2.5-7B.json", "AlphaEdit": "Qwen2.5-7B.json"}
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
}


def needs_transpose_for_svd(model_name):
    return model_name == "gpt2-xl"


def create_name_database(ds_name="mcf", limit=2000):
    subjects, _ = load_dataset_data(ds_name=ds_name, limit=limit)
    return subjects


def get_project(model, tok, layer, hparams):

    cov = alphaedit_get_cov(
        model, tok, hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype,
        force_recompute=False,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    small = (S < hparams.nullspace_threshold).nonzero(as_tuple=True)[0]
    return U[:, small] @ U[:, small].T


def get_activation_vector_for_name(model, tok, name, template, layer, module_template):
    activation_input, _ = get_module_input_output_at_words(
        model=model, tok=tok, layer=layer,
        context_templates=[template], words=[name],
        module_template=module_template, fact_token_strategy="subject_last",
    )
    return activation_input.squeeze() if activation_input.dim() > 1 else activation_input


def precompute_all_k_cands(model, tok, name_database, knowledge_template, layer, module_template):

    print(f"Precomputing k_cands for {len(name_database)} subjects...")
    k_cands = {}
    for name in name_database:
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).detach().cpu()
            k_cands[name] = k_cand
        except Exception:
            continue
    print(f"Precomputed {len(k_cands)} valid k_cands.")
    return k_cands


def compute_metrics(scores, true_subjects, top_k):
    if not scores or not true_subjects:
        return 0.0, 0.0
    top_names = {x["name"] for x in scores[:top_k]}
    recall = sum(1 for s in true_subjects if s in top_names) / len(true_subjects)
    score_dict = {x["name"]: x["score"] for x in scores}
    avg_true_score = float(np.mean([score_dict.get(s, 0.0) for s in true_subjects]))
    return recall, avg_true_score


def run_sequential_attack(delta_W, k_cands_dict, cumulative_edits, true_subjects, alg_name, P_attack=None,
                          model_name=""):

    if needs_transpose_for_svd(model_name):
        delta_W = delta_W.T

    U, S, Vh = torch.linalg.svd(delta_W.double(), full_matrices=False)
    rank = min(cumulative_edits, Vh.shape[0])
    Q_basis = Vh[:rank, :].T.cuda()

    scores = []
    for name, k_cand in k_cands_dict.items():
        k_cand = k_cand.cuda().double()


        if alg_name == "AlphaEdit" and P_attack is not None:
            v_test = P_attack @ k_cand
            proj_coeffs = Q_basis.T @ v_test
            score = torch.norm(proj_coeffs) / (torch.norm(v_test) + 1e-9)

        else:
            proj_coeffs = Q_basis.T @ k_cand
            score = torch.norm(proj_coeffs) / (torch.norm(k_cand) + 1e-9)

        scores.append({"name": name, "score": score.item()})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return compute_metrics(scores, true_subjects, cumulative_edits)


def run_single_sequential_experiment(alg_name, model_name, ds_name, num_edits, sequential_nums, run_id, results_dir):
    print("\n" + "=" * 80)
    print(
        f"Sequential Run #{run_id} | {alg_name} | {model_name} | {ds_name} | batch={num_edits} | steps={sequential_nums}")
    print("=" * 80)

    # 1. Load model
    model_full_name = MODEL_NAME_MAP[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_full_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_full_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2. Load Hparams
    hparams_fname = HPARAMS_FILE_MAP[model_name][alg_name]
    if alg_name == "MEMIT":
        hparams = MEMITHyperParams.from_json(HPARAMS_DIR / "MEMIT" / hparams_fname)
    elif alg_name == "AlphaEdit":
        hparams = AlphaEditHyperParams.from_json(HPARAMS_DIR / "AlphaEdit" / hparams_fname)

    # 3. Load Data & Create Chunks
    from dsets import MultiCounterFactDataset, MENDQADataset
    if ds_name == "mcf":
        ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
    elif ds_name == "zsre":
        ds = MENDQADataset(DATA_DIR, tok=tok, size=2000)

    ds_list = list(ds)
    total_edits = num_edits * sequential_nums

    random.seed(run_id * 2024)
    np.random.seed(run_id * 2024)
    sampled_records = random.sample(ds_list, total_edits)
    chunks = [sampled_records[i * num_edits:(i + 1) * num_edits] for i in range(sequential_nums)]

    # 4. Attack Prep (Initial weights, P_attack, K_cands)
    layer, module_template = ATTACK_CONFIG_MAP[(model_name, alg_name)]
    weight_name = f"{module_template.format(layer)}.weight"
    initial_weight = nethook.get_parameter(model, weight_name).detach().to("cuda").double().clone()

    knowledge_template = "The mother tongue of {} is"
    name_database = create_name_database(ds_name=ds_name)


    k_cands_dict = precompute_all_k_cands(model, tok, name_database, knowledge_template, layer, module_template)

    P_attack = None
    cache_c = None
    P_edit = None
    if alg_name == "AlphaEdit":
        print("Initializing AlphaEdit states...")
        P_attack = get_project(model, tok, layer, hparams).cuda().double()

        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        d_idx = 0 if hparams.model_name == "gpt2-xl" else 1
        d_shape = W_out.shape[d_idx]

        cache_c = torch.zeros((len(hparams.layers), d_shape, d_shape), device="cpu")
        P_edit = torch.zeros((len(hparams.layers), d_shape, d_shape), device="cpu")
        for i, lyr in enumerate(hparams.layers):
            P_edit[i, :, :] = get_project(model, tok, lyr, hparams)

    # 5. Execute Sequential Edits
    rows = []
    cumulative_subjects = []

    for step_idx, chunk in enumerate(chunks, start=1):
        print(f"\n--- Step {step_idx}/{sequential_nums} ---")
        edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in chunk]
        chunk_subjects = [r["requested_rewrite"]["subject"] for r in chunk]
        cumulative_subjects.extend(chunk_subjects)


        if alg_name == "MEMIT":
            model, _ = apply_memit_to_model(
                model=model, tok=tok, requests=edit_data, hparams=hparams,
                copy=False, return_orig_weights=False,
            )
        elif alg_name == "AlphaEdit":
            model, cache_c = apply_AlphaEdit_to_model(
                model=model, tok=tok, requests=edit_data, hparams=hparams,
                cache_template=None, cache_c=cache_c, P=P_edit,
            )


        cur_weight = nethook.get_parameter(model, weight_name).detach().to("cuda").double()
        delta_W = cur_weight - initial_weight

        cumulative_edits_count = step_idx * num_edits

        if step_idx >=2:

            recall, avg_true_score = run_sequential_attack(
                delta_W=delta_W,
                k_cands_dict=k_cands_dict,
                cumulative_edits=cumulative_edits_count,
                true_subjects=cumulative_subjects,
                alg_name=alg_name,
                P_attack=P_attack,
                model_name=model_name
            )

            row_data = {
                "Run_ID": run_id,
                "Model": model_name,
                "Algorithm": alg_name,
                "Dataset": ds_name,
                "Step": step_idx,
                "Cumulative_Edits": cumulative_edits_count,
                "Recall": recall,
                "Avg_Score": avg_true_score
            }
            rows.append(row_data)
            print(f"Cumulative Edits: {cumulative_edits_count} | Recall: {recall:.4f} | Avg Score: {avg_true_score:.6f}")


    del model
    del k_cands_dict
    torch.cuda.empty_cache()

    return rows


def generate_summary_tables(df, out_dir):

    df_summary = (
        df.groupby(["Model", "Algorithm", "Dataset", "Step", "Cumulative_Edits"])
        .agg(
            Recall_Mean=("Recall", "mean"),
            Recall_Std=("Recall", "std"),
            Score_Mean=("Avg_Score", "mean"),
            Score_Std=("Avg_Score", "std")
        )
        .reset_index()
    )

    df_summary["Recall_Fmt"] = df_summary.apply(lambda r: f"{r['Recall_Mean']:.4f} ± {r['Recall_Std']:.4f}", axis=1)
    df_summary["Score_Fmt"] = df_summary.apply(lambda r: f"{r['Score_Mean']:.6f} ± {r['Score_Std']:.6f}", axis=1)


    pivot_recall = df_summary.pivot(index=["Model", "Algorithm", "Dataset"], columns="Cumulative_Edits",
                                    values="Recall_Fmt")
    pivot_score = df_summary.pivot(index=["Model", "Algorithm", "Dataset"], columns="Cumulative_Edits",
                                   values="Score_Fmt")

    excel_path = out_dir / "sequential_attack_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw_Details", index=False)
        pivot_recall.to_excel(writer, sheet_name="Recall_Decline")
        pivot_score.to_excel(writer, sheet_name="Score_Decline")

    print(f"\nResults saved to: {excel_path}")
    print("\n--- Recall Decline Summary ---")
    print(pivot_recall.to_string())


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_edits", type=int, default=10)
    parser.add_argument("--sequential_nums", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    models = ["gpt-j", "Qwen2.5", "Llama3"]
    algorithms = ["MEMIT", "AlphaEdit"]
    datasets = ["mcf", "zsre"]

    # models = ["Llama3"]
    # algorithms = ["AlphaEdit"]
    # datasets = ["mcf"]

    out_dir = Path("additional_experiments/sequential_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name in models:
        for alg_name in algorithms:
            for ds_name in datasets:
                for run_id in range(args.runs):
                    try:
                        rows = run_single_sequential_experiment(
                            alg_name=alg_name,
                            model_name=model_name,
                            ds_name=ds_name,
                            num_edits=args.num_edits,
                            sequential_nums=args.sequential_nums,
                            run_id=run_id,
                            results_dir=out_dir
                        )
                        all_results.extend(rows)


                        temp_csv_path = out_dir / "temp_sequential_details.csv"
                        temp_json_path = out_dir / "temp_sequential_details.json"

                        pd.DataFrame(all_results).to_csv(temp_csv_path, index=False)
                        with open(temp_json_path, "w", encoding="utf-8") as f:
                            json.dump(all_results, f, indent=4, ensure_ascii=False)

                    except Exception as e:
                        print(f"Error in {model_name}-{alg_name}-{ds_name} Run {run_id}: {e}")
                        import traceback
                        traceback.print_exc()


    if all_results:
        df = pd.DataFrame(all_results)
        generate_summary_tables(df, out_dir)


if __name__ == "__main__":
    main()