"""
分析 prompt_recovery_results 目录下的 intermediate_results.json，
生成 Top-1、Top-5、Top-20 百分比统计表格

基于 prompt_recovery_experiment.py 的实验配置和数据结构
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_intermediate_results(results_dir):
    """
    加载 intermediate_results.json 文件
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        dict: 解析后的结果字典，key 为 (model, alg, ds, num_edits, run_id)
    """
    intermediate_path = results_dir / "intermediate_results.json"
    
    if not intermediate_path.exists():
        raise FileNotFoundError(f"File not found: {intermediate_path}")
    
    with open(intermediate_path, "r") as f:
        serializable_results = json.load(f)
    
    # 将字符串 key 转换回元组 key
    # 格式: {model}_{alg}_{ds}_{num_edits}_{run_id}
    # 注意：model 名称可能包含连字符（如 "gpt2-xl"），但不包含下划线
    # 所以 split("_") 会正确分割成 5 个部分
    
    all_results = {}
    for key_str, value in serializable_results.items():
        parts = key_str.split("_")
        
        # 格式应该是: {model}_{alg}_{ds}_{num_edits}_{run_id}
        # 所以应该有 5 个部分
        if len(parts) != 5:
            print(f"Warning: Key '{key_str}' does not have exactly 5 parts (got {len(parts)}), skipping...")
            continue
        
        try:
            # 直接解析：parts[0]=model, parts[1]=alg, parts[2]=ds, parts[3]=num_edits, parts[4]=run_id
            model = parts[0]
            alg = parts[1]
            ds = parts[2]
            num_edits = int(parts[3])
            run_id = int(parts[4])
            
            key = (model, alg, ds, num_edits, run_id)
            all_results[key] = value
                    
        except (ValueError, IndexError) as e:
            print(f"Error parsing key '{key_str}': {e}")
            print(f"  Parts: {parts}")
            continue
    
    return all_results


def calculate_top_n_percentage(ranks, n):
    """
    计算排名在 top-n 的百分比
    
    Args:
        ranks: 排名列表
        n: top-n 阈值
        
    Returns:
        float: 百分比值（0-100）
    """
    if not ranks:
        return 0.0
    
    # 过滤掉 None 值（编辑失败的情况）
    valid_ranks = [r for r in ranks if r is not None]
    if not valid_ranks:
        return 0.0
    
    # 统计排名 <= n 的数量
    top_n_count = sum(1 for rank in valid_ranks if rank <= n)
    percentage = (top_n_count / len(valid_ranks)) * 100.0
    
    return percentage


def generate_top_n_table(all_results, models, algorithms, datasets, num_edits_list, results_dir, n, table_name):
    """
    生成 Top-N 百分比表格
    
    Args:
        all_results: 所有实验结果字典
        models: 模型列表
        algorithms: 算法列表
        datasets: 数据集列表
        num_edits_list: 编辑数量列表
        results_dir: 结果目录
        n: top-n 阈值（1, 5, 或 20）
        table_name: 表格文件名（不含扩展名）
    """
    n_runs = 5  # 统一为 5 次独立实验
    
    table_data = []
    
    for model in models:
        for alg in algorithms:
            for ds in datasets:
                for num_edits in num_edits_list:
                    # 收集该配置下所有 run 的百分比
                    run_percentages = []
                    
                    for run_id in range(n_runs):
                        key = (model, alg, ds, num_edits, run_id)
                        
                        if key not in all_results:
                            continue
                        
                        results = all_results[key]
                        
                        # 从 summary 中获取 all_ranks
                        if "_summary" in results and "all_ranks" in results["_summary"]:
                            ranks = results["_summary"]["all_ranks"]
                            percentage = calculate_top_n_percentage(ranks, n)
                            run_percentages.append(percentage)
                    
                    # 计算均值和标准差
                    if run_percentages:
                        mean_percentage = np.mean(run_percentages)
                        std_percentage = np.std(run_percentages, ddof=1) if len(run_percentages) > 1 else 0.0
                    else:
                        mean_percentage = 0.0
                        std_percentage = 0.0
                    
                    table_data.append({
                        "Model": model,
                        "Algorithm": alg,
                        "Dataset": ds,
                        "Num_Edits": num_edits,
                        "N_runs": n_runs,
                        f"Top-{n} (%)": f"{mean_percentage:.1f}±{std_percentage:.1f}",
                    })
    
    df = pd.DataFrame(table_data)
    
    # 保存为 CSV
    csv_path = results_dir / f"{table_name}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{table_name} saved to: {csv_path}")
    print(f"\nPreview of {table_name}:")
    print(df.to_string(index=False))
    
    return df


def main():
    """主函数"""
    print("=" * 80)
    print("Generating Prompt Attack Statistics Tables")
    print("=" * 80)
    current_script_dir = Path(__file__).resolve().parent
    # 结果目录
    # results_dir = Path("main_ablation_experiments/prompt_recovery_results")
    results_dir = current_script_dir / "prompt_recovery_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # 加载中间结果
    print("\n[Step 1] Loading intermediate_results.json...")
    all_results = load_intermediate_results(results_dir)
    print(f"Loaded {len(all_results)} experiment results")
    
    # 从结果中提取所有唯一的配置（用于验证）
    unique_models = set()
    unique_algs = set()
    unique_ds = set()
    unique_num_edits = set()
    
    for key in all_results.keys():
        model, alg, ds, num_edits, run_id = key
        unique_models.add(model)
        unique_algs.add(alg)
        unique_ds.add(ds)
        unique_num_edits.add(num_edits)
    
    # 根据 prompt_recovery_experiment.py 的配置
    # 但实际应该从结果中推断，或者使用配置
    # 这里我们使用从结果中提取的配置
    models = sorted(list(unique_models))
    algorithms = sorted(list(unique_algs))
    datasets = sorted(list(unique_ds))
    num_edits_list = sorted(list(unique_num_edits))
    
    print(f"\nDetected configuration:")
    print(f"  Models: {models}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Datasets: {datasets}")
    print(f"  Num_edits: {num_edits_list}")
    
    # 生成三个表格
    print("\n[Step 2] Generating Top-1 percentage table...")
    generate_top_n_table(
        all_results, models, algorithms, datasets, num_edits_list,
        results_dir, n=1, table_name="table_top1_percentage"
    )
    
    print("\n[Step 3] Generating Top-5 percentage table...")
    generate_top_n_table(
        all_results, models, algorithms, datasets, num_edits_list,
        results_dir, n=5, table_name="table_top5_percentage"
    )
    
    print("\n[Step 4] Generating Top-20 percentage table...")
    generate_top_n_table(
        all_results, models, algorithms, datasets, num_edits_list,
        results_dir, n=20, table_name="table_top20_percentage"
    )
    
    print("\n" + "=" * 80)
    print("All tables generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

