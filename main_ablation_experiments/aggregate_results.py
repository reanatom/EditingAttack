

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def extract_scale_from_dirname(dirname: str) -> float:

    try:

        if dirname.startswith("camouflage_scale="):
            scale_str = dirname.replace("camouflage_scale=", "").strip("'\"")
            return float(scale_str)
    except:
        pass
    return None


def count_edit_dirs(alg_dir: Path) -> int:

    if not alg_dir.exists():
        return 0
    edit_dirs = [d for d in alg_dir.iterdir() if d.is_dir() and d.name.startswith("edit")]
    return len(edit_dirs)


def load_privacy_data(alg_dir: Path) -> Tuple[float, float]:

    privacy_file = alg_dir / "privacy_rank.json"
    if not privacy_file.exists():
        return (np.nan, np.nan)
    
    try:
        with open(privacy_file, 'r') as f:
            data = json.load(f)
        

        if "individual_run_avg_ranks" in data and data["individual_run_avg_ranks"]:
            ranks = data["individual_run_avg_ranks"]
            mean_rank = np.mean(ranks)
            std_rank = np.std(ranks)
            return (mean_rank, std_rank)
        

        avg_rank = data.get("overall_average_rank", np.nan)
        return (avg_rank, 0.0)
        
    except Exception as e:
        print(f"Error loading {privacy_file}: {e}")
        return (np.nan, np.nan)


def load_glue_scores(alg_dir: Path, datasets: List[str]) -> Dict[str, Tuple[float, float]]:


    dataset_scores = {ds: [] for ds in datasets}
    
    edit_dirs = sorted([d for d in alg_dir.iterdir() if d.is_dir() and d.name.startswith("edit")])
    
    for edit_dir in edit_dirs:
        glue_file = edit_dir / "glue_results.json"
        if not glue_file.exists():
            continue
        
        try:
            with open(glue_file, 'r') as f:
                data = json.load(f)
            

            for ds in datasets:
                if ds in data and isinstance(data[ds], dict) and "f1" in data[ds]:
                    f1_score = data[ds]["f1"]
                    dataset_scores[ds].append(f1_score)
        except Exception as e:
            print(f"Error loading {glue_file}: {e}")
    

    results = {}
    for ds in datasets:
        if dataset_scores[ds]:
            mean_score = np.mean(dataset_scores[ds])
            std_score = np.std(dataset_scores[ds])
            results[ds] = (mean_score, std_score)
        else:
            results[ds] = (np.nan, np.nan)
    
    return results


def aggregate_results():

    
    results_dir = Path("main_ablation_experiments/results")
    # results_dir = Path("main_ablation_experiments/results/zsre")
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("="*80)
    print("Aggregating Camouflage Scale Ablation Results")
    print("="*80)
    

    scale_dirs = []
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.startswith("camouflage_scale"):
            scale = extract_scale_from_dirname(d.name)
            if scale is not None:
                scale_dirs.append((scale, d))
    
    if not scale_dirs:
        print("Error: No camouflage_scale directories found")
        return
    

    scale_dirs.sort(key=lambda x: x[0])
    scales = [s for s, _ in scale_dirs]
    
    print(f"Found scales: {scales}")
    

    first_scale_dir = scale_dirs[0][1]
    # alg_dirs = [d.name for d in first_scale_dir.iterdir() if d.is_dir()]
    alg_dirs = ["ROME_defence"]
    alg_names = sorted(alg_dirs)
    
    print(f"Found algorithms: {alg_names}")
    

    n_runs = 0
    for alg_name in alg_names:
        alg_dir = first_scale_dir / alg_name
        n_runs = count_edit_dirs(alg_dir)
        if n_runs > 0:
            break
    
    print(f"Number of independent runs per scale: {n_runs}")
    
    if n_runs == 0:
        print("Error: No edit directories found")
        return
    

    print("\nGenerating Privacy Table...")
    privacy_data = []
    
    for scale, scale_dir in scale_dirs:
        row = {"camouflage_scale": scale}
        for alg_name in alg_names:
            alg_dir = scale_dir / alg_name
            mean_rank, std_rank = load_privacy_data(alg_dir)
            

            if not np.isnan(mean_rank):
                formatted = f"{mean_rank:.4f} ± {std_rank:.4f}"
            else:
                formatted = "N/A"
            
            row[alg_name] = formatted
        privacy_data.append(row)
    
    privacy_df = pd.DataFrame(privacy_data)
    privacy_df = privacy_df.set_index("camouflage_scale")
    privacy_filename = f"main_ablation_experiments/privacy_runs={n_runs}.xlsx"

    # privacy_filename = f"main_ablation_experiments/privacy_runs_zsre={n_runs}.xlsx"
    privacy_df.to_excel(privacy_filename)
    print(f"Privacy table saved to: {privacy_filename}")
    print(privacy_df)
    

    print("\nGenerating Effect Table...")
    

    datasets = ["sst", "mmmlu", "mrpc", "cola", "rte", "nli"]
    

    effect_data = []
    
    for scale, scale_dir in scale_dirs:
        row_data = {"camouflage_scale": scale}
        
        for alg_name in alg_names:
            alg_dir = scale_dir / alg_name
            glue_scores = load_glue_scores(alg_dir, datasets)
            

            for ds in datasets:
                mean, std = glue_scores[ds]

                if not np.isnan(mean):
                    formatted = f"{mean:.4f} ± {std:.4f}"
                else:
                    formatted = "N/A"
                

                col_name = f"{ds}_{alg_name}"
                row_data[col_name] = formatted
        
        effect_data.append(row_data)
    

    effect_df = pd.DataFrame(effect_data)
    effect_df = effect_df.set_index("camouflage_scale")
    

    multi_columns = []
    for ds in datasets:
        for alg_name in alg_names:
            multi_columns.append((ds, alg_name))
    

    reordered_data = []
    for scale, scale_dir in scale_dirs:
        row = []
        for ds in datasets:
            for alg_name in alg_names:
                alg_dir = scale_dir / alg_name
                glue_scores = load_glue_scores(alg_dir, datasets)
                mean, std = glue_scores[ds]
                if not np.isnan(mean):
                    formatted = f"{mean:.4f} ± {std:.4f}"
                else:
                    formatted = "N/A"
                row.append(formatted)
        reordered_data.append(row)
    

    multi_index = pd.MultiIndex.from_tuples(multi_columns, names=["Dataset", "Algorithm"])
    effect_df_multi = pd.DataFrame(reordered_data, columns=multi_index, index=scales)
    effect_df_multi.index.name = "camouflage_scale"
    

    effect_filename = f"main_ablation_experiments/effect_runs={n_runs}.xlsx"
    # effect_filename = f"main_ablation_experiments/effect_runs_zsre={n_runs}.xlsx"
    effect_df_multi.to_excel(effect_filename)
    print(f"\nEffect table saved to: {effect_filename}")
    print(effect_df_multi)
    
    print("\n" + "="*80)
    print("Aggregation Complete!")
    print("="*80)
    print(f"Privacy Table: {privacy_filename}")
    print(f"Effect Table: {effect_filename}")


if __name__ == "__main__":
    aggregate_results()

