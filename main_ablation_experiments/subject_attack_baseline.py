"""
Subject Attack Baseline: 基于JS散度的编辑主语推断攻击

该脚本作为 main_attack_experiment.py 的基线对比，使用 JS 散度来判断哪些主语被编辑。

逻辑：
1. 获取编辑前模型 (Pre-Model) 和编辑后模型 (Post-Model)。
2. 使用通用模板 "The mother tongue of {} is"。
3. 遍历候选主语库，分别输入 Pre-Model 和 Post-Model，获取 Next Token 分布。
4. 计算两个分布的 JS 散度。
5. 选取 JS 散度变化最大的 N 个主语（N=num_edits）作为推测的编辑对象。
6. 计算召回率。

实验设置与 main_attack_experiment.py 保持一致。
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
import torch.nn.functional as F
import gc

# Add project path
sys.path.append(os.getcwd())

# Import editing algorithms
from memit.memit_main import apply_memit_to_model
from memit.memit_hparams import MEMITHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov as alphaedit_get_cov
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from rome.rome_main import apply_rome_to_model
from rome.rome_hparams import ROMEHyperParams
from util import nethook
from util.data_loader import load_dataset_data
from util.globals import HPARAMS_DIR, DATA_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Model name mapping
MODEL_NAME_MAP = {
    "gpt2-xl": "gpt2-xl",
    "gpt-j": "EleutherAI/gpt-j-6b",
    "Llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen2.5": "Qwen/Qwen2.5-7B-Instruct"
}

# Hyperparameter file mapping
HPARAMS_FILE_MAP = {
    "gpt2-xl": {
        "MEMIT": "gpt2-xl.json",
        "AlphaEdit": "gpt2-xl.json",
        "ROME": "gpt2-xl.json"
    },
    "gpt-j": {
        "MEMIT": "EleutherAI_gpt-j-6B.json",
        "AlphaEdit": "EleutherAI_gpt-j-6B.json",
        "ROME": "EleutherAI_gpt-j-6B.json"
    },
    "Llama3": {
        "MEMIT": "Llama3-8B.json",
        "AlphaEdit": "Llama3-8B.json",
        "ROME": "Llama3-8B.json"
    },
    "Qwen2.5":{
        "MEMIT": "Qwen2.5-7B.json",
        "AlphaEdit": "Qwen2.5-7B.json",
        "ROME": "Qwen2.5-7B.json"
    }
}

def create_name_database(ds_name="mcf", limit=2000):
    """
    创建包含候选主语的数据库
    """
    subjects, _ = load_dataset_data(ds_name=ds_name, limit=limit)
    return subjects

def get_project(model, tok, layer, hparams):
    """
    计算投影矩阵P（用于AlphaEdit）
    """
    force_recompute = False
    cov = alphaedit_get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    # print(f"Layer {layer}: {len(small_singular_indices)} small singular values")
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T

def get_next_token_probs(model, tok, subjects, template, batch_size=64):
    """
    获取一系列主语在给定模板下的 Next Token 概率分布
    
    Args:
        model: 模型
        tok: Tokenizer
        subjects: 主语列表
        template: 提示词模板
        batch_size: 批次大小
        
    Returns:
        Tensor of shape (num_subjects, vocab_size) on CPU
    """
    model.eval()
    prompts = [template.format(s) for s in subjects]
    
    # 确保 tokenizer 设置了 pad_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize
            inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            outputs = model(**inputs)
            # 获取最后一个 token 的 logits
            # 对于 causal LM，inputs.input_ids[:, -1] 对应的输出 logits 预测的是下一个 token
            # 注意：如果使用 left padding，最后一个 token 就在 -1。如果是 right padding，需要找到真实长度。
            # 这里简化处理，假设 padding 处理得当，或者我们直接取 -1 (通常 inputs['attention_mask'] 会被模型处理)
            # 更严谨的做法是取 attention_mask 最后一个 1 的位置，但在 batch inference 中通常 -1 是 padding (right pad) 或者 最后一个词 (left pad)。
            # AutoTokenizer 默认通常是 right padding。但是 model(**inputs) 会处理 mask。
            # 如果是 right padding，-1 位置可能是 pad token 的输出。
            # 为了保险，我们需要取出每个序列最后一个有效 token 的 logits。
            
            logits = outputs.logits
            
            # 获取每个样本最后一个有效 token 的索引
            # attention_mask: (batch, seq_len)
            last_token_indices = inputs.attention_mask.sum(dim=1) - 1
            
            # 提取对应的 logits: (batch, vocab_size)
            batch_logits = logits[torch.arange(logits.size(0)), last_token_indices]
            
            probs = F.softmax(batch_logits, dim=-1).cpu()
            all_probs.append(probs)
            
    return torch.cat(all_probs, dim=0)

def calculate_js_divergence(p, q):
    """
    计算两个分布 P 和 Q 之间的 JS 散度
    P, Q: (num_subjects, vocab_size), 已经是概率分布 (sum=1)
    """
    # 加上极小值防止 log(0)
    epsilon = 1e-9
    
    # M = 0.5 * (P + Q)
    m = 0.5 * (p + q)
    
    # KL(P || M) = sum(P * log(P / M))
    kl_p_m = (p * (torch.log(p + epsilon) - torch.log(m + epsilon))).sum(dim=1)
    
    # KL(Q || M) = sum(Q * log(Q / M))
    kl_q_m = (q * (torch.log(q + epsilon) - torch.log(m + epsilon))).sum(dim=1)
    
    # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    js_div = 0.5 * kl_p_m + 0.5 * kl_q_m
    
    return js_div

def run_single_experiment(
    alg_name,
    model_name,
    hparams_fname,
    ds_name,
    num_edits,
    run_id,
    results_dir
):
    print("\n" + "=" * 80)
    print(f"Experiment Run #{run_id}")
    print(f"Algorithm: {alg_name}, Model: {model_name}, Dataset: {ds_name}, Num Edits: {num_edits}")
    print("=" * 80)
    
    # 1. Load model and tokenizer
    print("\n[Step 1] Loading model...")
    model_full_name = MODEL_NAME_MAP[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_full_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_full_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # 2. Load hyperparameters
    if alg_name == "MEMIT":
        params_path = HPARAMS_DIR / "MEMIT" / hparams_fname
        hparams = MEMITHyperParams.from_json(params_path)
    elif alg_name == "AlphaEdit":
        params_path = HPARAMS_DIR / "AlphaEdit" / hparams_fname
        hparams = AlphaEditHyperParams.from_json(params_path)
    elif alg_name == "ROME":
        params_path = HPARAMS_DIR / "ROME" / hparams_fname
        hparams = ROMEHyperParams.from_json(params_path)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    
    # 3. Load dataset
    print("\n[Step 2] Loading dataset...")
    from dsets import MultiCounterFactDataset, MENDQADataset
    
    if ds_name == "mcf":
        ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
    elif ds_name == "zsre":
        ds = MENDQADataset(DATA_DIR, tok=tok, size=2000)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    
    ds_list = list(ds)
    
    # Sample num_edits samples
    np.random.seed(run_id * 12345)
    random.seed(run_id * 12345)
    sampled_records = random.sample(ds_list, num_edits)
    
    # Extract true subjects
    true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]
    
    # Prepare request format
    if alg_name == "ROME":
        edit_data = [{"case_id": sampled_records[0]["case_id"], **sampled_records[0]["requested_rewrite"]}]
    else:
        edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]
    
    print(f"Selected {len(edit_data)} samples for editing")
    
    # 4. Compute Pre-edit Probabilities (Baseline Attack Step 1)
    print("\n[Step 3] Computing Pre-edit probabilities...")
    knowledge_template = "The city {} lives in is"
    name_database = create_name_database(ds_name=ds_name)
    
    # 获取 Pre-edit 分布 (CPU Tensor to save VRAM)
    pre_edit_probs = get_next_token_probs(model, tok, name_database, knowledge_template)
    
    # 5. Execute editing
    print("\n[Step 4] Executing editing...")
    
    try:
        if alg_name == "MEMIT":
            edited_model, _ = apply_memit_to_model(
                model=model,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                copy=False, # In-place or copy=False logic
                return_orig_weights=False,
            )
        elif alg_name == "AlphaEdit":
            # Initialize cache_c and P for AlphaEdit
            print("Initializing cache_c and P for AlphaEdit...")
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
            
            # 使用与 main_attack_experiment.py 相同的逻辑初始化 P
            if hparams.model_name == "gpt2-xl":
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            elif hparams.model_name in ["EleutherAI_gpt-j-6B", "Llama3-8B", "phi-1.5"]:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            else:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            
            del W_out
            
            print("Computing projection matrices P for each layer...")
            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(model, tok, layer, hparams)
            
            edited_model, cache_c = apply_AlphaEdit_to_model(
                model=model,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                cache_template=None,
                cache_c=cache_c,
                P=P,
            )
        elif alg_name == "ROME":
            edited_model, _ = apply_rome_to_model(
                model=model,
                tok=tok,
                request=edit_data,
                hparams=hparams,
                copy=False,
                return_orig_weights=False,
            )
        
        print("Editing completed!")
    except Exception as e:
        print(f"Error in editing: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup
        del model
        torch.cuda.empty_cache()
        return np.nan
    
    # 6. Compute Post-edit Probabilities (Baseline Attack Step 2)
    print("\n[Step 5] Computing Post-edit probabilities...")
    # 注意：如果 copy=False，model 和 edited_model 可能是同一个对象。
    # 之前的 pre_edit_probs 已经在 Step 3 计算并保存在 CPU 上了，所以这里直接用 edited_model 计算即可。
    post_edit_probs = get_next_token_probs(edited_model, tok, name_database, knowledge_template)
    
    # 7. Compute JS Divergence (Baseline Attack Step 3)
    print("\n[Step 6] Calculating JS Divergence...")
    js_scores = calculate_js_divergence(pre_edit_probs, post_edit_probs)
    
    # 8. Rank and Calculate Recall
    print("\n[Step 7] Calculating Recall...")
    scored_subjects = []
    for i, name in enumerate(name_database):
        scored_subjects.append({
            "name": name,
            "score": js_scores[i].item()
        })
    
    # 降序排列 (JS散度越大，表示变化越大，越可能是编辑目标)
    scored_subjects.sort(key=lambda x: x["score"], reverse=True)
    
    # 取前 num_edits 个作为预测结果
    top_n_names = {item["name"] for item in scored_subjects[:num_edits]}
    
    recall_count = sum(1 for subj in true_subjects if subj in top_n_names)
    recall_rate = recall_count / len(true_subjects) if true_subjects else 0.0
    
    print(f"Baseline Recall Rate: {recall_rate:.4f}")
    
    # Cleanup
    del model
    del edited_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return recall_rate

def generate_tables(all_results, models, algorithms, datasets, num_edit_settings, results_dir, suffix=""):
    """生成结果表"""
    recall_data = []
    for model in models:
        for alg in algorithms:
            for ds in datasets:
                row = {"Model": model, "Algorithm": alg, "Dataset": ds}
                for num_edit in num_edit_settings:
                    key = (model, alg, ds, num_edit)
                    values = all_results.get(key, [])
                    
                    valid_values = [v for v in values if not np.isnan(v)]
                    
                    if valid_values:
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                        row[f"Num={num_edit}"] = f"{mean_val:.4f} ± {std_val:.4f}"
                    else:
                        row[f"Num={num_edit}"] = "N/A"
                
                recall_data.append(row)
    
    df_recall = pd.DataFrame(recall_data)
    
    # Save
    excel_path = results_dir / f"baseline_attack_results{suffix}.xlsx"
    df_recall.to_excel(excel_path, index=False)
    
    csv_path = results_dir / f"baseline_attack_results{suffix}.csv"
    df_recall.to_csv(csv_path, index=False)
    
    print(f"\nTables saved to {excel_path} and {csv_path}")
    print("\nTable Preview:")
    print(df_recall.to_string(index=False))

def run_multi_edit_experiments():
    """运行多编辑实验"""
    print("\n" + "=" * 80)
    print("Multi-Edit Baseline Attack Experiments")
    print("=" * 80)
    
    models = ["Llama3", "gpt-j"]
    num_edit_settings = [10, 50, 100]
    algorithms = ["MEMIT","AlphaEdit"]
    datasets = ["mcf", "zsre"]
    # models = ["Llama3"]
    # num_edit_settings = [10,50,100]
    # algorithms = ["MEMIT"]
    # datasets = ["mcf","zsre"]
    n_independent_runs = 5
    
    results_dir = Path("main_ablation_experiments/baseline_attack_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {} # Key: (model, alg, ds, num_edit), Value: list of recalls
    
    total_experiments = len(models) * len(num_edit_settings) * len(algorithms) * len(datasets) * n_independent_runs
    experiment_count = 0
    
    for model_name in models:
        for alg_name in algorithms:
            for ds_name in datasets:
                for num_edits in num_edit_settings:
                    key = (model_name, alg_name, ds_name, num_edits)
                    recalls = []
                    
                    print("\n" + "=" * 80)
                    print(f"Configuration: Model={model_name}, Algorithm={alg_name}, Dataset={ds_name}, Num Edits={num_edits}")
                    print("=" * 80)
                    
                    hparams_fname = HPARAMS_FILE_MAP[model_name][alg_name]
                    
                    for run_id in range(n_independent_runs):
                        experiment_count += 1
                        print(f"\nProgress: {experiment_count}/{total_experiments}")
                        
                        try:
                            recall = run_single_experiment(
                                alg_name=alg_name,
                                model_name=model_name,
                                hparams_fname=hparams_fname,
                                ds_name=ds_name,
                                num_edits=num_edits,
                                run_id=run_id,
                                results_dir=results_dir,
                            )
                            recalls.append(recall)
                        except Exception as e:
                            print(f"Error: Experiment failed - {e}")
                            import traceback
                            traceback.print_exc()
                            recalls.append(np.nan)
                    
                    all_results[key] = recalls
                    
                    # Intermediate save
                    with open(results_dir / "multi_edit_intermediate_increment.json", "w") as f:
                        str_key_results = {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in all_results.items()}
                        json.dump(str_key_results, f, indent=2)

    generate_tables(all_results, models, algorithms, datasets, num_edit_settings, results_dir, suffix="_multi")

def run_single_edit_experiments():
    """运行单编辑实验 (ROME)"""
    print("\n" + "=" * 80)
    print("Single-Edit Baseline Attack Experiments (ROME)")
    print("=" * 80)
    
    models = ["gpt-j", "Llama3", "Qwen2.5"]
    algorithms = ["ROME"]
    datasets = ["mcf", "zsre"]
    num_edits = 1
    n_independent_runs = 5
    
    results_dir = Path("main_ablation_experiments/baseline_attack_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {} # Key: (model, alg, ds, num_edit), Value: list of recalls
    
    total_experiments = len(models) * len(datasets) * n_independent_runs
    experiment_count = 0
    
    for model_name in models:
        for alg_name in algorithms:
            for ds_name in datasets:
                key = (model_name, alg_name, ds_name, num_edits)
                recalls = []
                
                print("\n" + "=" * 80)
                print(f"Configuration: Model={model_name}, Algorithm={alg_name}, Dataset={ds_name}")
                print("=" * 80)
                
                hparams_fname = HPARAMS_FILE_MAP[model_name][alg_name]
                
                for run_id in range(n_independent_runs):
                    experiment_count += 1
                    print(f"\nProgress: {experiment_count}/{total_experiments}")
                    
                    try:
                        recall = run_single_experiment(
                            alg_name=alg_name,
                            model_name=model_name,
                            hparams_fname=hparams_fname,
                            ds_name=ds_name,
                            num_edits=num_edits,
                            run_id=run_id,
                            results_dir=results_dir,
                        )
                        recalls.append(recall)
                    except Exception as e:
                        print(f"Error: Experiment failed - {e}")
                        import traceback
                        traceback.print_exc()
                        recalls.append(np.nan)
                
                all_results[key] = recalls
                
                # Intermediate save
                with open(results_dir / "single_edit_intermediate.json", "w") as f:
                    str_key_results = {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in all_results.items()}
                    json.dump(str_key_results, f, indent=2)

    generate_tables(all_results, models, algorithms, datasets, [1], results_dir, suffix="_single")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_type",
        choices=["multi", "single", "both"],
        default="both",
        help="Type of experiment to run: multi-edit, single-edit, or both"
    )
    args = parser.parse_args()
    
    if args.experiment_type in ["multi", "both"]:
        run_multi_edit_experiments()
    
    if args.experiment_type in ["single", "both"]:
        run_single_edit_experiments()
        
    print("\n" + "=" * 80)
    print("All baseline experiments completed!")
    print("=" * 80)
