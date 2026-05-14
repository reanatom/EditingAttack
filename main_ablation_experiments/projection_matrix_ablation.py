"""
Projection Matrix Estimation Ablation Study (AlphaEdit)

Test the impact of covariance matrices (used for Projection Matrix P) computed with different datasets and sample sizes on AlphaEdit subject attack effectiveness.

Experiment Settings:
- Datasets: wikipedia, wikitext, pile
- Sample sizes: 10, 100, 1000, 10000
- Number of edits: 100
- Independent runs: 5
- Evaluation metrics: Top-100 recall rate, average projection score
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from copy import deepcopy

# Add project path
sys.path.append(os.getcwd())

# Import AlphaEdit specifics
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov as get_cov_standard
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from memit.compute_z import get_module_input_output_at_words
from dsets import MultiCounterFactDataset
from util.globals import HPARAMS_DIR, DATA_DIR
from util import nethook
from rome.layer_stats import layer_stats

# Global cache for covariance stats
COV_CACHE = {}

# Use estimated stats directory
STATS_DIR_ESTIMATED = Path("data/stats_estimated")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_cov_estimated(
    model,
    tok,
    layer_name,
    mom2_dataset,
    mom2_n_samples,
    mom2_dtype,
    inv=False,
    force_recompute=False,
):
    """
    Retrieves covariance statistics from STATS_DIR_ESTIMATED.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name, mom2_dataset, mom2_n_samples)

    print(f"Retrieving estimated covariance statistics for {model_name} @ {layer_name} using {mom2_dataset} ({mom2_n_samples} samples).")
    
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR_ESTIMATED, # Use the specific estimated stats dir
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

def get_project_estimated(model, tok, layer, hparams, cov_dataset, cov_sample_size):
    """
    Compute Projection Matrix P using estimated covariance statistics.
    """
    force_recompute = False
    
    # Use the ablation parameters instead of hparams for dataset and sample size
    cov = get_cov_estimated(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        cov_dataset, # Use ablation dataset
        cov_sample_size, # Use ablation sample size
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    # print(f"Layer {layer}: {len(small_singular_indices)} small singular values")
    
    # P = U_null @ U_null.T
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T

def create_name_database():
    from util.data_loader import load_dataset_data
    subjects, _ = load_dataset_data(limit=2000)
    return subjects

def get_activation_vector_for_name(
    model, tok, name, template, layer, module_template, fact_token_strategy
):
    context_templates = [template]
    words = [name]
    
    activation_input, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy=fact_token_strategy,
    )
    
    if activation_input.dim() > 1:
        activation_input = activation_input.squeeze()
    
    return activation_input

def perform_alphaedit_attack(model, tok, config, true_subjects, knowledge_template, P_matrix):
    """
    Perform AlphaEdit attack (Rank-N Subspace + P projection)
    """
    print(f"\n[Performing AlphaEdit Subject Attack]")
    
    # 1. Load Edit Amounts (Delta)
    edit_amounts_path = config.delta_file_path
    if not Path(edit_amounts_path).exists():
        print(f"Error: File not found: {edit_amounts_path}")
        return 0.0, 0.0
        
    try:
        edit_amounts = torch.load(edit_amounts_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return 0.0, 0.0
        
    # Get Delta for the target layer
    layer_key = f"{config.rewrite_module_tmp.format(config.final_layer_to_attack)}.weight"
    if layer_key not in edit_amounts:
        print(f"Error: Layer {layer_key} not found in edit amounts.")
        return 0.0, 0.0
        
    layer_data = edit_amounts[layer_key]
    if isinstance(layer_data, dict):
        delta = layer_data["delta"].cuda().double()
    else:
        delta = layer_data.cuda().double()
        
    # 2. SVD to get Subspace Q
    try:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        actual_rank = min(config.target_rank, Vh.shape[0])
        Q_basis = Vh[:actual_rank, :].T # (In, N)
    except Exception as e:
        print(f"SVD failed: {e}")
        return 0.0, 0.0

    # 3. Score Candidates
    name_database = create_name_database()
    scores = []
    
    Q_basis = Q_basis.to("cuda").double()
    P_matrix = P_matrix.to("cuda").double()
    
    print(f"Computing projection scores for {len(name_database)} candidate subjects...")
    
    for i, name in enumerate(name_database):
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template,
                config.final_layer_to_attack,
                config.rewrite_module_tmp,
                "subject_last"
            ).to("cuda").double()
            
            # Project using ESTIMATED P: v_test = P @ k_cand
            v_test = P_matrix @ k_cand
            
            # Projection Score = || Q^T v_test || / || v_test ||
            proj_coeffs = Q_basis.T @ v_test
            proj_norm = torch.norm(proj_coeffs)
            test_norm = torch.norm(v_test)
            
            score = proj_norm / (test_norm + 1e-9)
            
            scores.append({
                "name": name,
                "score": score.item()
            })
            
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1}/{len(name_database)}")
            
        except Exception as e:
             # print(f"Warning: Error processing {name} - {e}")
             pass

    scores.sort(key=lambda x: x["score"], reverse=True)
    
    top_100_names = {item["name"] for item in scores[:100]}
    
    recall_count = sum(1 for subj in true_subjects if subj in top_100_names)
    recall_at_100 = recall_count / len(true_subjects)
    
    score_dict = {item["name"]: item["score"] for item in scores}
    true_scores = [score_dict.get(subj, 0.0) for subj in true_subjects]
    avg_proj_score = np.mean(true_scores) if true_scores else 0.0
    
    print(f"Top-100 Recall Rate: {recall_at_100:.4f} ({recall_count}/{len(true_subjects)})")
    print(f"Average Projection Score: {avg_proj_score:.6f}")
    
    return recall_at_100, avg_proj_score

def run_single_experiment(
    model_name,
    base_hparams,
    num_edits,
    run_id,
    cov_dataset,
    cov_sample_size,
    results_dir
):
    print("\n" + "=" * 80)
    print(f"Experiment #{run_id}: Covariance={cov_dataset}, Size={cov_sample_size}")
    
    # 1. Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    # 2. Data
    print("Preparing data...")
    ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
    ds_list = list(ds)
    random.seed(run_id * 12345)
    np.random.seed(run_id * 12345)
    sampled_records = random.sample(ds_list, num_edits)
    true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]
    edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]
    
    # 3. Pre-compute AlphaEdit dependencies (Standard P for Editing)
    print("Initializing AlphaEdit dependencies (Standard)...")
    W_out = nethook.get_parameter(model, f"{base_hparams.rewrite_module_tmp.format(base_hparams.layers[-1])}.weight")
    
    if base_hparams.model_name in ["EleutherAI_gpt-j-6B", "Llama3-8B", "phi-1.5"]:
        size = W_out.shape[1]
    else:
        size = W_out.shape[0] # gpt2-xl
        
    cache_c = torch.zeros((len(base_hparams.layers), size, size), device="cpu")
    P_edit = torch.zeros((len(base_hparams.layers), size, size), device="cpu")
    
    def get_project_standard(model, tok, layer, hparams):
        cov = get_cov_standard(
            model, tok, hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype
        ).cpu()
        U, S, _ = torch.linalg.svd(cov, full_matrices=False)
        small_singular_indices = (S < hparams.nullspace_threshold).nonzero(as_tuple=True)[0]
        return U[:, small_singular_indices] @ U[:, small_singular_indices].T

    print("Computing standard P matrices for editing...")
    for i, layer in enumerate(base_hparams.layers):
        P_edit[i,:,:] = get_project_standard(model, tok, layer, base_hparams)
        
    del W_out

    # 4. Execute Editing
    print("Executing AlphaEdit...")
    apply_AlphaEdit_to_model(
        model=model,
        tok=tok,
        requests=edit_data,
        hparams=base_hparams,
        cache_c=cache_c,
        P=P_edit
    )
    
    # 5. Prepare Attack P Matrix (Estimated)
    target_layer = 4 # Using layer 4 for attack as in covariance_ablation
    print(f"Computing Attack P Matrix for layer {target_layer} using {cov_dataset}, {cov_sample_size}...")
    
    P_attack = get_project_estimated(
        model, tok, target_layer, base_hparams, cov_dataset, cov_sample_size
    )
    
    # 6. Attack Config
    case_id = edit_data[0]["case_id"]
    attack_config = SimpleNamespace(
        delta_file_path=f"alphaedit_edit_amounts/edit_amounts_batch_case_{case_id}.pt",
        rewrite_module_tmp=base_hparams.rewrite_module_tmp,
        final_layer_to_attack=target_layer,
        target_rank=num_edits
    )
    
    # 7. Attack
    # Use ORIGINAL model for attack (load new instance)
    print("Reloading original model for attack...")
    del model
    torch.cuda.empty_cache()
    
    original_model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    
    recall, proj_score = perform_alphaedit_attack(
        original_model, tok, attack_config, true_subjects, 
        "The mother tongue of {} is", P_attack
    )
    
    del original_model
    del P_attack
    del P_edit
    del cache_c
    torch.cuda.empty_cache()
    
    print(f"Experiment #{run_id} completed!")
    return recall, proj_score

def generate_result_tables(all_results, cov_datasets, cov_sample_sizes, results_dir):
    """
    Generate result tables and save as Excel
    """
    # Table 1: Top-100 Recall Rate
    recall_data = []
    for dataset in cov_datasets:
        row = {"Dataset": dataset}
        for size in cov_sample_sizes:
            key = (dataset, size)
            values = all_results["recall"].get(key, [])
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                row[f"Size={size}"] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[f"Size={size}"] = "N/A"
        recall_data.append(row)
    
    df_recall = pd.DataFrame(recall_data)
    
    # Table 2: Average Projection Score
    proj_score_data = []
    for dataset in cov_datasets:
        row = {"Dataset": dataset}
        for size in cov_sample_sizes:
            key = (dataset, size)
            values = all_results["proj_score"].get(key, [])
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                row[f"Size={size}"] = f"{mean_val:.6f} ± {std_val:.6f}"
            else:
                row[f"Size={size}"] = "N/A"
        proj_score_data.append(row)
    
    df_proj_score = pd.DataFrame(proj_score_data)
    
    # Save as Excel
    excel_path = results_dir / "projection_matrix_ablation_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_recall.to_excel(writer, sheet_name='Top100_Recall', index=False)
        df_proj_score.to_excel(writer, sheet_name='Avg_Projection_Score', index=False)
    
    print(f"\nExcel table saved: {excel_path}")
    
    # Print table preview
    print("\n" + "=" * 80)
    print("Table 1: Top-100 Recall Rate")
    print("=" * 80)
    print(df_recall.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Table 2: Average Projection Score")
    print("=" * 80)
    print(df_proj_score.to_string(index=False))

def run_ablation_study():
    """
    Run complete projection matrix ablation study
    """
    print("\n" + "=" * 80)
    print("Projection Matrix Estimation Ablation Study (AlphaEdit)")
    print("=" * 80)
    
    # Experiment configuration
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_edits = 100
    n_independent_runs = 5
    
    # Covariance matrix estimation settings for ablation
    # cov_datasets = ["wikipedia", "wikitext", "pile"]
    # cov_sample_sizes = [10000, 1000, 100, 10]

    cov_datasets = ["pile", "wikitext"]
    cov_sample_sizes = [10, 100, 1000, 10000]

    # Results directory
    results_dir = Path("main_ablation_experiments/projection_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base hyperparameters
    hparams_fname = "Llama3-8B.json"
    params_path = HPARAMS_DIR / "AlphaEdit" / hparams_fname
    base_hparams = AlphaEditHyperParams.from_json(params_path)
    print(f"Loaded base parameters from {params_path}")
    
    # Store all results
    all_results = {
        "recall": {}, 
        "proj_score": {}
    }
    
    # Main loop
    total_experiments = len(cov_datasets) * len(cov_sample_sizes) * n_independent_runs
    experiment_count = 0
    
    for cov_dataset in cov_datasets:
        for cov_sample_size in cov_sample_sizes:
            key = (cov_dataset, cov_sample_size)
            recalls = []
            proj_scores = []
            
            print("\n" + "=" * 80)
            print(f"Estimation Setting: Dataset={cov_dataset}, Sample Size={cov_sample_size}")
            print("=" * 80)
            
            for run_id in range(n_independent_runs):
                experiment_count += 1
                print(f"\nProgress: {experiment_count}/{total_experiments}")
                
                try:
                    recall, proj_score = run_single_experiment(
                        model_name=model_name,
                        base_hparams=base_hparams,
                        num_edits=num_edits,
                        run_id=run_id,
                        cov_dataset=cov_dataset,
                        cov_sample_size=cov_sample_size,
                        results_dir=results_dir,
                    )
                    
                    recalls.append(recall)
                    proj_scores.append(proj_score)
                    
                except Exception as e:
                    print(f"Error: Experiment failed - {e}")
                    import traceback
                    traceback.print_exc()
                    
                    recalls.append(np.nan)
                    proj_scores.append(np.nan)
            
            # Save results
            all_results["recall"][key] = recalls
            all_results["proj_score"][key] = proj_scores
            
            # Intermediate save
            intermediate_path = results_dir / "intermediate_results_increment.json"
            with open(intermediate_path, "w") as f:
                serializable_results = {
                    "recall": {f"{k[0]}_{k[1]}": v for k, v in all_results["recall"].items()},
                    "proj_score": {f"{k[0]}_{k[1]}": v for k, v in all_results["proj_score"].items()}
                }
                json.dump(serializable_results, f, indent=2)
            print(f"\nIntermediate results saved to: {intermediate_path}")
    
    # Generate final tables
    generate_result_tables(all_results, cov_datasets, cov_sample_sizes, results_dir)
    print("\nExperiment completed! Results saved to:", results_dir)

if __name__ == "__main__":
    run_ablation_study()

