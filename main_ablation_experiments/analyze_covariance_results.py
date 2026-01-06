"""
分析协方差消融实验的结果，生成可视化图表
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置字体为 Times New Roman（与论文一致）
plt.rcParams['font.family'] = 'Times New Roman'


def load_results(results_dir):
    """
    从中间结果文件加载数据
    """
    intermediate_path = Path(results_dir) / "intermediate_results.json"
    
    if not intermediate_path.exists():
        print(f"错误: 找不到结果文件 {intermediate_path}")
        return None
    
    with open(intermediate_path, "r") as f:
        data = json.load(f)
    
    # 转换回原格式
    results = {
        "recall": {},
        "proj_score": {}
    }
    
    for key_str, values in data["recall"].items():
        dataset, size = key_str.rsplit("_", 1)
        results["recall"][(dataset, int(size))] = values
    
    for key_str, values in data["proj_score"].items():
        dataset, size = key_str.rsplit("_", 1)
        results["proj_score"][(dataset, int(size))] = values
    
    return results


def print_detailed_stats(results, cov_datasets, cov_sample_sizes):
    """
    打印详细统计信息
    """
    print("\n" + "=" * 80)
    print("详细统计信息")
    print("=" * 80)
    
    for metric_name, metric_data in [("Top-100召回率", "recall"), ("平均投影分数", "proj_score")]:
        print(f"\n{metric_name}:")
        print("-" * 80)
        
        for dataset in cov_datasets:
            print(f"\n数据集: {dataset}")
            for size in cov_sample_sizes:
                key = (dataset, size)
                values = results[metric_data].get(key, [])
                
                valid_values = [v for v in values if not np.isnan(v)]
                
                if valid_values:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                    min_val = np.min(valid_values)
                    max_val = np.max(valid_values)
                    
                    print(f"  Size={size:5d}: {mean_val:.6f} ± {std_val:.6f} "
                          f"[{min_val:.6f}, {max_val:.6f}] (n={len(valid_values)})")
                else:
                    print(f"  Size={size:5d}: N/A (没有有效数据)")


def create_heatmaps(results, cov_datasets, cov_sample_sizes, results_dir):
    """
    创建热力图可视化
    """
    print("\n生成热力图...")
    
    # 准备数据矩阵
    recall_matrix = np.zeros((len(cov_datasets), len(cov_sample_sizes)))
    proj_score_matrix = np.zeros((len(cov_datasets), len(cov_sample_sizes)))
    
    for i, dataset in enumerate(cov_datasets):
        for j, size in enumerate(cov_sample_sizes):
            key = (dataset, size)
            
            # 召回率
            recall_values = [v for v in results["recall"].get(key, []) if not np.isnan(v)]
            recall_matrix[i, j] = np.mean(recall_values) if recall_values else np.nan
            
            # 投影分数
            proj_values = [v for v in results["proj_score"].get(key, []) if not np.isnan(v)]
            proj_score_matrix[i, j] = np.mean(proj_values) if proj_values else np.nan
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 热力图1: 召回率
    sns.heatmap(
        recall_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        xticklabels=[f'{s}' for s in cov_sample_sizes],
        yticklabels=cov_datasets,
        cbar_kws={'label': 'Recall Rate'},
        ax=ax1,
        vmin=0.0,
        vmax=1.0
    )
    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('Dataset', fontsize=12)
    ax1.set_title('Top-100 Recall Rate', fontsize=14, fontweight='bold')
    
    # 热力图2: 投影分数
    sns.heatmap(
        proj_score_matrix,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        xticklabels=[f'{s}' for s in cov_sample_sizes],
        yticklabels=cov_datasets,
        cbar_kws={'label': 'Projection Score'},
        ax=ax2
    )
    ax2.set_xlabel('Sample Size', fontsize=12)
    ax2.set_ylabel('Dataset', fontsize=12)
    ax2.set_title('Average Projection Score', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    heatmap_path = Path(results_dir) / "covariance_heatmaps.pdf"
    plt.savefig(heatmap_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"热力图已保存: {heatmap_path}")
    
    plt.close()


def create_line_plots(results, cov_datasets, cov_sample_sizes, results_dir):
    """
    创建折线图（按数据集分组）
    """
    print("\n生成折线图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 折线图1: 召回率
    for i, dataset in enumerate(cov_datasets):
        means = []
        stds = []
        for size in cov_sample_sizes:
            key = (dataset, size)
            values = [v for v in results["recall"].get(key, []) if not np.isnan(v)]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            else:
                means.append(np.nan)
                stds.append(0.0)
        
        means = np.array(means)
        stds = np.array(stds)
        
        ax1.plot(cov_sample_sizes, means, marker=markers[i], label=dataset,
                 color=colors[i], linewidth=2, markersize=8)
        ax1.fill_between(cov_sample_sizes, means - stds, means + stds,
                          alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('Recall Rate', fontsize=12)
    ax1.set_title('Top-100 Recall vs Sample Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 折线图2: 投影分数
    for i, dataset in enumerate(cov_datasets):
        means = []
        stds = []
        for size in cov_sample_sizes:
            key = (dataset, size)
            values = [v for v in results["proj_score"].get(key, []) if not np.isnan(v)]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            else:
                means.append(np.nan)
                stds.append(0.0)
        
        means = np.array(means)
        stds = np.array(stds)
        
        ax2.plot(cov_sample_sizes, means, marker=markers[i], label=dataset,
                 color=colors[i], linewidth=2, markersize=8)
        ax2.fill_between(cov_sample_sizes, means - stds, means + stds,
                          alpha=0.2, color=colors[i])
    
    ax2.set_xlabel('Sample Size', fontsize=12)
    ax2.set_ylabel('Projection Score', fontsize=12)
    ax2.set_title('Projection Score vs Sample Size', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    lineplot_path = Path(results_dir) / "covariance_lineplots.pdf"
    plt.savefig(lineplot_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"折线图已保存: {lineplot_path}")
    
    plt.close()


def main():
    results_dir = Path("main_ablation_experiments/covariance_results")
    
    if not results_dir.exists():
        print(f"错误: 结果目录不存在 - {results_dir}")
        return
    
    print("加载实验结果...")
    results = load_results(results_dir)
    
    if results is None:
        return
    
    cov_datasets = ["wikipedia", "wikitext", "pile"]
    cov_sample_sizes = [10, 100, 1000, 10000]
    
    # 打印详细统计
    print_detailed_stats(results, cov_datasets, cov_sample_sizes)
    
    # 生成可视化
    create_heatmaps(results, cov_datasets, cov_sample_sizes, results_dir)
    create_line_plots(results, cov_datasets, cov_sample_sizes, results_dir)
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()

