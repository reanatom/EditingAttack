"""
聚合 Camouflage Scale 消融实验的详细指标
统计 metrics: rewrite_prompts_correct, paraphrase_prompts_correct, neighborhood_prompts_correct, ngram_entropy, reference_score
逻辑：
1. 遍历 main_ablation_experiments/results/camouflage_scale={scale}/{alg_name}/edit* (直接在results下一级查找)
2. 读取目录下所有 *_edits-case_*.json
3. 计算 metrics (mean over cases for each run)
4. 统计 runs (mean ± std)
5. 输出表格
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import glob
import re

# METRICS = [
#     "rewrite_prompts_correct",
#     "paraphrase_prompts_correct",
#     "neighborhood_prompts_correct",
#     "ngram_entropy",
#     "reference_score"
# ]

METRICS = [
    "rewrite_prompts_correct",
    "paraphrase_prompts_correct",
    "neighborhood_prompts_correct",
    # "ngram_entropy",
    # "reference_score"
]

def extract_scale_from_dirname(dirname: str) -> float:
    """从目录名提取 scale 值"""
    try:
        # 格式: camouflage_scale=X
        if dirname.startswith("camouflage_scale="):
            scale_str = dirname.replace("camouflage_scale=", "").strip("'\"")
            return float(scale_str)
    except:
        pass
    return None

def calculate_metric_score(val):
    """根据 value 类型计算得分"""
    if val is None:
        return np.nan
    
    if isinstance(val, list):
        if len(val) == 0:
            return 0.0
        # 布尔列表，True=1, False=0
        # 有时可能是 [True, False] 或 [1, 0]
        numeric_list = [1.0 if x else 0.0 for x in val]
        return np.mean(numeric_list)
    
    # 直接数值
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def process_edit_run(edit_dir: Path) -> Dict[str, float]:
    """处理单个 edit 目录，返回该 run 中所有 cases 的 metric 均值"""
    
    # 查找符合模式的 json 文件
    # 模式: *_edits-case_*.json
    json_files = list(edit_dir.glob("*_edits-case_*.json"))
    
    if not json_files:
        return {}
    
    # 存储该 run 中每个 case 的得分
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
            
    # 计算该 run 的平均分 (Mean)
    run_averages = {}
    for metric in METRICS:
        if run_scores[metric]:
            run_averages[metric] = np.mean(run_scores[metric])
        else:
            run_averages[metric] = np.nan
            
    return run_averages

def aggregate_detailed_metrics():
    # results_dir = Path("main_ablation_experiments/results")
    results_dir = Path("main_ablation_experiments/results/zsre")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("="*80)
    print("Aggregating Detailed Metrics for Camouflage Scale Ablation")
    print("="*80)

    # 1. 扫描 scale 目录
    scale_dirs = []
    
    # 用户要求：只看 results 里面的目录，不看 zsre 的
    # 直接遍历 results_dir 下的一级子目录，不进入子文件夹（如 zsre）
    
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.startswith("camouflage_scale="):
            scale = extract_scale_from_dirname(d.name)
            if scale is not None:
                scale_dirs.append((scale, d))
    
    if not scale_dirs:
        print("Error: No 'camouflage_scale=*' directories found directly under main_ablation_experiments/results/")
        # 提示用户可能数据在子目录但被忽略了
        print("Note: Subdirectories (like 'zsre/') were skipped as per instructions.")
        return
        
    print(f"Found {len(scale_dirs)} directories matching 'camouflage_scale=*' in results root.")
    
    scale_dirs.sort(key=lambda x: x[0])
    scales = sorted(list(set([s for s, _ in scale_dirs])))
    print(f"Found scales: {scales}")

    # 2. 获取所有算法名 (从所有 scale 目录中收集)
    all_algs = set()
    for _, s_dir in scale_dirs:
        for item in s_dir.iterdir():
            if item.is_dir():
                all_algs.add(item.name)
    alg_names = sorted(list(all_algs))
    print(f"Found algorithms: {alg_names}")

    # 数据结构: data[scale][alg][metric] = (mean, std)
    summary_data = []

    for scale, scale_dir in scale_dirs:
        for alg in alg_names:
            alg_dir = scale_dir / alg
            if not alg_dir.exists():
                continue
            
            # 找到所有 edit* 目录
            edit_dirs = sorted([d for d in alg_dir.iterdir() if d.is_dir() and d.name.startswith("edit")])
            
            if not edit_dirs:
                continue
                
            # 收集每个 run 的结果
            # run_results[metric] = [score_run1, score_run2, ...]
            run_results = {m: [] for m in METRICS}
            
            for ed in edit_dirs:
                run_avg = process_edit_run(ed)
                for m in METRICS:
                    if m in run_avg and not np.isnan(run_avg[m]):
                        run_results[m].append(run_avg[m])
            
            # 计算 Mean +/- Std
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

    # 创建 DataFrame
    df = pd.DataFrame(summary_data)
    
    # 设置多级索引方便查看 (Scale, Algorithm)
    if not df.empty:
        df = df.set_index(["camouflage_scale", "Algorithm"])
        
        # 输出到 Excel 和 CSV
        # output_xlsx = "main_ablation_experiments/detailed_metrics_ablation.xlsx"
        # output_csv = "main_ablation_experiments/detailed_metrics_ablation.csv"

        output_xlsx = "main_ablation_experiments/zsre_detailed_metrics_ablation.xlsx"
        output_csv = "main_ablation_experiments/zsre_detailed_metrics_ablation.csv"
        
        df.to_excel(output_xlsx)
        df.to_csv(output_csv)
        
        print("\nAggregation Results:")
        print(df)
        print(f"\nSaved detailed metrics to:\n  {output_xlsx}\n  {output_csv}")
    else:
        print("\nNo data found to aggregate.")

if __name__ == "__main__":
    aggregate_detailed_metrics()
