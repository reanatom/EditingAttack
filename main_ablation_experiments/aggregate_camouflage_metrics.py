

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import glob
import re

METRICS = [
    "rewrite_prompts_correct",
    "paraphrase_prompts_correct",
    "neighborhood_prompts_correct",
    "ngram_entropy",
    "reference_score"
]

# METRICS = [
#     "rewrite_prompts_correct",
#     "paraphrase_prompts_correct",
#     "neighborhood_prompts_correct",
#     # "ngram_entropy",
#     # "reference_score"
# ]

def extract_scale_from_dirname(dirname: str) -> float:

    try:

        if dirname.startswith("camouflage_scale="):
            scale_str = dirname.replace("camouflage_scale=", "").strip("'\"")
            return float(scale_str)
    except:
        pass
    return None

def calculate_metric_score(val):

    if val is None:
        return np.nan
    
    if isinstance(val, list):
        if len(val) == 0:
            return 0.0

        numeric_list = [1.0 if x else 0.0 for x in val]
        return np.mean(numeric_list)
    

    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def process_edit_run(edit_dir: Path) -> Dict[str, float]:

    

    json_files = list(edit_dir.glob("*_edits-case_*.json"))
    
    if not json_files:
        return {}
    

    run_scores = {m: [] for m in METRICS}
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            post_data = data.get("post", {})
            
            for metric in METRICS:
                val = post_data.get(metric)
                score = calculate_metric_score(val)
                if not np.isnan(score):
                    run_scores[metric].append(score)
                    
        except Exception as e:
            print(f"Error reading {jf}: {e}")
            continue
            

    run_averages = {}
    for metric in METRICS:
        if run_scores[metric]:
            run_averages[metric] = np.mean(run_scores[metric])
        else:
            run_averages[metric] = np.nan
            
    return run_averages

def aggregate_detailed_metrics():
    results_dir = Path("main_ablation_experiments/results")
    # results_dir = Path("main_ablation_experiments/results/zsre")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("="*80)
    print("Aggregating Detailed Metrics for Camouflage Scale Ablation")
    print("="*80)


    scale_dirs = []
    

    
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.startswith("camouflage_scale="):
            scale = extract_scale_from_dirname(d.name)
            if scale is not None:
                scale_dirs.append((scale, d))
    
    if not scale_dirs:
        print("Error: No 'camouflage_scale=*' directories found directly under main_ablation_experiments/results/")

        print("Note: Subdirectories (like 'zsre/') were skipped as per instructions.")
        return
        
    print(f"Found {len(scale_dirs)} directories matching 'camouflage_scale=*' in results root.")
    
    scale_dirs.sort(key=lambda x: x[0])
    scales = sorted(list(set([s for s, _ in scale_dirs])))
    print(f"Found scales: {scales}")


    all_algs = set()
    for _, s_dir in scale_dirs:
        for item in s_dir.iterdir():
            if item.is_dir():
                all_algs.add(item.name)

    alg_names = sorted(list(all_algs))

    alg_names =["ROME_defence"]

    print(f"Found algorithms: {alg_names}")


    summary_data = []

    for scale, scale_dir in scale_dirs:
        for alg in alg_names:
            alg_dir = scale_dir / alg
            if not alg_dir.exists():
                continue
            

            edit_dirs = sorted([d for d in alg_dir.iterdir() if d.is_dir() and d.name.startswith("edit")])
            
            if not edit_dirs:
                continue
                

            # run_results[metric] = [score_run1, score_run2, ...]
            run_results = {m: [] for m in METRICS}
            
            for ed in edit_dirs:
                run_avg = process_edit_run(ed)
                for m in METRICS:
                    if m in run_avg and not np.isnan(run_avg[m]):
                        run_results[m].append(run_avg[m])
            

            row = {
                "camouflage_scale": scale,
                "Algorithm": alg
            }
            
            for m in METRICS:
                scores = run_results[m]
                if scores:
                    mean = np.mean(scores)
                    std = np.std(scores)
                    row[m] = f"{mean:.4f} ± {std:.4f}"
                else:
                    row[m] = "N/A"
            
            summary_data.append(row)


    df = pd.DataFrame(summary_data)
    

    if not df.empty:
        df = df.set_index(["camouflage_scale", "Algorithm"])
        

        output_xlsx = "main_ablation_experiments/detailed_metrics_ablation.xlsx"
        output_csv = "main_ablation_experiments/detailed_metrics_ablation.csv"

        # output_xlsx = "main_ablation_experiments/zsre_detailed_metrics_ablation.xlsx"
        # output_csv = "main_ablation_experiments/zsre_detailed_metrics_ablation.csv"
        
        df.to_excel(output_xlsx)
        df.to_csv(output_csv)
        
        print("\nAggregation Results:")
        print(df)
        print(f"\nSaved detailed metrics to:\n  {output_xlsx}\n  {output_csv}")
    else:
        print("\nNo data found to aggregate.")

if __name__ == "__main__":
    aggregate_detailed_metrics()
