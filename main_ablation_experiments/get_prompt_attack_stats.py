

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_intermediate_results(results_dir):

    intermediate_path = results_dir / "intermediate_results.json"
    
    if not intermediate_path.exists():
        raise FileNotFoundError(f"File not found: {intermediate_path}")
    
    with open(intermediate_path, "r") as f:
        serializable_results = json.load(f)
    

    
    all_results = {}
    for key_str, value in serializable_results.items():
        parts = key_str.split("_")
        

        if len(parts) != 5:
            print(f"Warning: Key '{key_str}' does not have exactly 5 parts (got {len(parts)}), skipping...")
            continue
        
        try:

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

    if not ranks:
        return 0.0
    

    valid_ranks = [r for r in ranks if r is not None]
    if not valid_ranks:
        return 0.0
    

    top_n_count = sum(1 for rank in valid_ranks if rank <= n)
    percentage = (top_n_count / len(valid_ranks)) * 100.0
    
    return percentage


def generate_top_n_table(all_results, models, algorithms, datasets, num_edits_list, results_dir, n, table_name):

    n_runs = 5
    
    table_data = []
    
    for model in models:
        for alg in algorithms:
            for ds in datasets:
                for num_edits in num_edits_list:

                    run_percentages = []
                    
                    for run_id in range(n_runs):
                        key = (model, alg, ds, num_edits, run_id)
                        
                        if key not in all_results:
                            continue
                        
                        results = all_results[key]
                        

                        if "_summary" in results and "all_ranks" in results["_summary"]:
                            ranks = results["_summary"]["all_ranks"]
                            percentage = calculate_top_n_percentage(ranks, n)
                            run_percentages.append(percentage)
                    

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
    

    csv_path = results_dir / f"{table_name}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{table_name} saved to: {csv_path}")
    print(f"\nPreview of {table_name}:")
    print(df.to_string(index=False))
    
    return df


def main():

    print("=" * 80)
    print("Generating Prompt Attack Statistics Tables")
    print("=" * 80)
    current_script_dir = Path(__file__).resolve().parent

    # results_dir = Path("main_ablation_experiments/prompt_recovery_results")
    results_dir = current_script_dir / "prompt_recovery_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    

    print("\n[Step 1] Loading intermediate_results.json...")
    all_results = load_intermediate_results(results_dir)
    print(f"Loaded {len(all_results)} experiment results")
    

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
    

    models = sorted(list(unique_models))
    algorithms = sorted(list(unique_algs))
    datasets = sorted(list(unique_ds))
    num_edits_list = sorted(list(unique_num_edits))
    
    print(f"\nDetected configuration:")
    print(f"  Models: {models}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Datasets: {datasets}")
    print(f"  Num_edits: {num_edits_list}")
    


    generate_top_n_table(
        all_results, models, algorithms, datasets, num_edits_list,
        results_dir, n=1, table_name="table_top1_percentage"
    )
    

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

