"""
Covariance Matrix Estimation Ablation Study

Test the impact of covariance matrices computed with different datasets and sample sizes on subject attack effectiveness.

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
from copy import deepcopy

# Add project path
sys.path.append(os.getcwd())

from memit.memit_main import apply_memit_to_model
from memit.compute_z import get_module_input_output_at_words
from memit.memit_hparams import MEMITHyperParams
from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT
from dsets import MultiCounterFactDataset
from util.globals import HPARAMS_DIR, DATA_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_name_database():
    """
    Create a database containing 2000 candidate subjects
    """
    from util.data_loader import load_dataset_data
    subjects, _ = load_dataset_data(limit=2000)
    return subjects


def get_activation_vector_for_name(
    model, tok, name, template, layer, module_template, fact_token_strategy
):
    """
    Get activation vector for a given name at target layer
    """
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


def perform_subject_attack(model, tok, config, true_subjects, knowledge_template):
    """
    Perform subject attack and return recall rate and projection score
    
    Returns:
        recall_at_100: Top-100 recall rate
        avg_proj_score: Average projection score of true subjects
    """
    print(f"\n[Performing Subject Attack]")
    
    # 1. Get subspace basis
    try:
        Q_basis = run_attack_simple_k_r_CORRECT(config, model, tok)
        print(f"Subspace basis shape: {Q_basis.shape}")
    except Exception as e:
        print(f"Error: Failed to recover subspace basis - {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0
    
    # 2. Build candidate database
    name_database = create_name_database()
    scores = []
    
    Q_basis = Q_basis.to("cuda").double()
    
    print(f"Computing projection scores for {len(name_database)} candidate subjects...")
    for i, name in enumerate(name_database):
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template,
                config.final_layer_to_attack,
                config.rewrite_module_tmp,
                "subject_last"
            ).to("cuda").double()
            
            # Compute projection score
            proj_coeffs = Q_basis.T @ k_cand
            proj_norm = torch.norm(proj_coeffs)
            cand_norm = torch.norm(k_cand)
            
            score = proj_norm / (cand_norm + 1e-9)
            
            scores.append({
                "name": name,
                "score": score.item(),
            })
            
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{len(name_database)}")
        
        except Exception as e:
            print(f"Warning: Error processing {name} - {e}")
    
    # 3. Sort
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    # 4. Compute metrics
    top_100_names = {item["name"] for item in scores[:100]}
    
    # Recall rate: proportion of true subjects in top 100
    recall_count = sum(1 for subj in true_subjects if subj in top_100_names)
    recall_at_100 = recall_count / len(true_subjects)
    
    # Average projection score: projection scores of true subjects
    score_dict = {item["name"]: item["score"] for item in scores}
    true_scores = [score_dict.get(subj, 0.0) for subj in true_subjects]
    avg_proj_score = np.mean(true_scores)
    
    print(f"Top-100 Recall Rate: {recall_at_100:.4f} ({recall_count}/{len(true_subjects)})")
    print(f"Average Projection Score: {avg_proj_score:.6f}")
    
    return recall_at_100, avg_proj_score


def run_single_experiment(
    model_name,
    base_hparams,
    num_edits,
    dataset_name,
    run_id,
    cov_dataset,
    cov_sample_size,
    results_dir
):
    """
    Run a single independent experiment: editing + attack
    
    Returns:
        recall_at_100: Top-100 recall rate
        avg_proj_score: Average projection score
    """
    print("\n" + "=" * 80)
    print(f"Independent Experiment #{run_id}")
    print(f"Covariance Estimation: Dataset={cov_dataset}, Sample Size={cov_sample_size}")
    print("=" * 80)
    
    # 1. Load model and tokenizer
    print("\n[Step 1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # 2. Prepare editing data
    print("\n[Step 2] Preparing editing data...")
    # Load data using MultiCounterFactDataset
    ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
    ds_list = list(ds)
    
    # Sample num_edits samples
    np.random.seed(run_id * 12345)  # Different seed for each independent experiment
    import random
    random.seed(run_id * 12345)
    sampled_records = random.sample(ds_list, num_edits)
    
    # Extract true subjects
    true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]
    
    # Prepare request format (consistent with camouflage_scale_ablation.py)
    edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]
    
    print(f"Selected {len(edit_data)} samples for editing")
    print(f"True subjects example: {true_subjects[:3]}")
    
    # 3. Execute MEMIT editing
    print("\n[Step 3] Executing MEMIT editing...")
    edited_model, _ = apply_memit_to_model(
        model=model,
        tok=tok,
        requests=edit_data,
        hparams=base_hparams,
        copy=False,
        return_orig_weights=False,
    )
    
    print("Editing completed!")
    
    # 4. Prepare attack configuration
    print("\n[Step 4] Preparing attack configuration...")
    # Use the first case_id (batch file uses the first sample's case_id)
    case_id = edit_data[0]["case_id"]
    
    attack_config = SimpleNamespace(
        model_name=model_name,
        delta_file_path=f"multi_case_edit_memit_amount/edit_amounts_batch_case_{case_id}.pt",
        kr_ground_truth_path=f"multi_case_edit_memit_amount/kr_ground_truth_case_{case_id}.pt",
        
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        final_layer_to_attack=4,
        
        target_rank=num_edits,  # Subspace rank equals number of edits
        
        # Key: Use specified covariance estimation settings
        mom2_dataset=cov_dataset,
        mom2_n_samples=cov_sample_size,
        mom2_dtype="float32",
        mom2_update_weight=15000,
        
        # Optimizer parameters (skipped when rank > 1)
        learning_rate_r=1e-2,
        iterations=0,
        beta=5e-6,
        lambda_l2=5e-6,
    )
    
    # Use unified prompt template
    knowledge_template = "The mother tongue of {} is"
    
    # 5. Load original model for attack (attack needs original model, not edited model)
    print("\n[Step 5] Loading original model for attack...")
    original_model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    
    # 6. Execute attack
    print("\n[Step 6] Executing subject attack...")
    recall, proj_score = perform_subject_attack(
        original_model, tok, attack_config, true_subjects, knowledge_template
    )
    
    # 7. Cleanup
    del model
    del edited_model
    del original_model
    torch.cuda.empty_cache()
    
    print(f"\nExperiment #{run_id} completed!")
    print(f"  Recall Rate: {recall:.4f}")
    print(f"  Projection Score: {proj_score:.6f}")
    
    return recall, proj_score


def run_ablation_study():
    """
    Run complete covariance ablation study
    """
    print("\n" + "=" * 80)
    print("Covariance Matrix Estimation Ablation Study")
    print("=" * 80)
    
    # Experiment configuration
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_edits = 100
    n_independent_runs = 5
    
    # Covariance matrix estimation settings
    cov_datasets = ["wikipedia", "wikitext", "pile"]
    cov_sample_sizes = [10, 100, 1000, 10000]
    
    # Results directory
    results_dir = Path("main_ablation_experiments/covariance_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base hyperparameters (consistent with camouflage_scale_ablation.py)
    hparams_fname = "Llama3-8B.json"
    params_path = HPARAMS_DIR / "MEMIT" / hparams_fname
    base_hparams = MEMITHyperParams.from_json(params_path)
    print(f"Loaded base parameters from {params_path}")
    
    # Store all results
    all_results = {
        "recall": {},  # {(dataset, size): [run1, run2, ...]}
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
            print(f"Covariance Estimation Setting: Dataset={cov_dataset}, Sample Size={cov_sample_size}")
            print("=" * 80)
            
            for run_id in range(n_independent_runs):
                experiment_count += 1
                print(f"\nProgress: {experiment_count}/{total_experiments}")
                
                try:
                    recall, proj_score = run_single_experiment(
                        model_name=model_name,
                        base_hparams=base_hparams,
                        num_edits=num_edits,
                        dataset_name="mcf",
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
                    
                    # Record failure
                    recalls.append(np.nan)
                    proj_scores.append(np.nan)
            
            # Save results for this configuration
            all_results["recall"][key] = recalls
            all_results["proj_score"][key] = proj_scores
            
            # Intermediate save
            intermediate_path = results_dir / "intermediate_results.json"
            with open(intermediate_path, "w") as f:
                # Convert to serializable format
                serializable_results = {
                    "recall": {f"{k[0]}_{k[1]}": v for k, v in all_results["recall"].items()},
                    "proj_score": {f"{k[0]}_{k[1]}": v for k, v in all_results["proj_score"].items()}
                }
                json.dump(serializable_results, f, indent=2)
            print(f"\nIntermediate results saved to: {intermediate_path}")
    
    # Generate final tables
    print("\n" + "=" * 80)
    print("Generating final result tables...")
    print("=" * 80)
    
    generate_result_tables(all_results, cov_datasets, cov_sample_sizes, results_dir)
    
    print("\nExperiment completed! Results saved to:", results_dir)


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
            
            # Filter NaN
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
            
            # Filter NaN
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
    excel_path = results_dir / "covariance_ablation_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_recall.to_excel(writer, sheet_name='Top100_Recall', index=False)
        df_proj_score.to_excel(writer, sheet_name='Avg_Projection_Score', index=False)
    
    print(f"\nExcel table saved: {excel_path}")
    
    # Also save as CSV (backup)
    df_recall.to_csv(results_dir / "recall_results.csv", index=False)
    df_proj_score.to_csv(results_dir / "proj_score_results.csv", index=False)
    
    # Print table preview
    print("\n" + "=" * 80)
    print("Table 1: Top-100 Recall Rate")
    print("=" * 80)
    print(df_recall.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Table 2: Average Projection Score")
    print("=" * 80)
    print(df_proj_score.to_string(index=False))


if __name__ == "__main__":
    run_ablation_study()

