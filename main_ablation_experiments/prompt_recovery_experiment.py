"""
提示词恢复攻击主实验：基于熵差恢复编辑使用的提示词

实验设置：
- 数据集：mcf（前2000条数据）
- 模型：gpt2-xl, gpt-j, Llama3
- 编辑数量：100个
- 独立重复实验：5次

指标：
1. Target恢复成功率：100个知识中恢复target_true的占比
2. 平均语义相似度：每个主语恢复的提示词和真实提示词的平均相似度，对所有主语求和取平均
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
import random

# Add project path
sys.path.append(os.getcwd())

# Import editing algorithms
from memit.memit_main import apply_memit_to_model
from memit.memit_hparams import MEMITHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov as alphaedit_get_cov
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from rome.rome_main import apply_rome_to_model
from rome.rome_hparams import ROMEHyperParams
from util.data_loader import load_dataset_data
from util.globals import HPARAMS_DIR, DATA_DIR, RESULTS_DIR
from util import nethook

# Import evaluation functions
from dsets import AttributeSnippets, get_tfidf_vectorizer
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre

# 尝试导入sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic similarity calculation will be disabled.")

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
        "ROME": "gpt2-xl.json",
    },
    "gpt-j": {
        "MEMIT": "EleutherAI_gpt-j-6B.json",
        "AlphaEdit": "EleutherAI_gpt-j-6B.json",
        "ROME": "EleutherAI_gpt-j-6B.json",
    },
    "Llama3": {
        "MEMIT": "Llama3-8B.json",
        "AlphaEdit": "Llama3-8B.json",
        "ROME": "Llama3-8B.json",
    },
    "Qwen2.5":{
        "MEMIT": "Qwen2.5-7B.json",
        "AlphaEdit": "Qwen2.5-7B.json",
        "ROME": "Qwen2.5-7B.json",
    }
}

# Dataset evaluation method mapping
DS_EVAL_METHOD_MAP = {
    "mcf": compute_rewrite_quality_counterfact,
    "zsre": compute_rewrite_quality_zsre,
}


def calculate_prediction_entropy_batch(model, tok, prompts, subjects, batch_size=32):
    """
    批量计算模型对下一个词预测的熵 (Entropy)
    
    Args:
        model: 模型
        tok: 分词器
        prompts: 提示词模板列表（包含{}占位符）
        subjects: 主语列表
        batch_size: 批次大小
    
    Returns:
        entropies: 熵值列表，与输入顺序对应
    """
    # 构建所有 subject-prompt 组合
    full_prompts = [prompt.format(subject) for prompt, subject in zip(prompts, subjects)]
    
    entropies = []
    
    # 分批处理
    for i in range(0, len(full_prompts), batch_size):
        batch_prompts = full_prompts[i:i + batch_size]
        
        # 批量编码，使用 padding
        inputs = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # 根据模型调整
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取每个样本最后一个 token 的 logits
        # logits shape: [batch_size, seq_len, vocab_size]
        batch_logits = outputs.logits
        
        # 获取每个样本最后一个非padding token的logits
        # 使用 attention_mask 找到最后一个有效token
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 因为索引从0开始
        
        batch_entropies = []
        for j in range(batch_logits.size(0)):
            last_token_idx = seq_lengths[j].item()
            logits = batch_logits[j, last_token_idx, :]
            
            # 计算熵: -sum(p * log(p))
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum().item()
            batch_entropies.append(entropy)
        
        entropies.extend(batch_entropies)
    
    return entropies


def load_sentence_model():
    """全局加载SentenceBERT模型，避免重复加载"""
    sentence_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            print("Loading SentenceBERT model...")
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceBERT model loaded successfully.")
        except Exception as e:
            print(f"Error loading SentenceBERT model: {e}")
            print("Semantic similarity calculation will be disabled.")
    
    return sentence_model


def get_project(model, tok, layer, hparams):
    """
    计算投影矩阵P（用于AlphaEdit）
    参考evaluate.py和main_attack_experiment.py中的实现
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
    print(f"Layer {layer}: {len(small_singular_indices)} small singular values")
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def calculate_rank_distribution(ranks):
    """
    计算排名分布统计
    
    Args:
        ranks: 排名列表
    
    Returns:
        dict: 包含不同区间范围的统计
    """
    # 定义区间范围
    intervals = [
        (1, 5),
        (5, 20),
        (20, 50),
        (50, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, 600),
        (600, 700),
        (700, 800),
        (800, 900),
        (900, 1000),
    ]
    
    distribution = {}
    for start, end in intervals:
        # 统计在 [start, end) 区间内的排名数量
        # 注意：区间是左闭右开的 [start, end)
        count = sum(1 for rank in ranks if start <= rank < end)
        distribution[f"{start}-{end}"] = count
    
    # 统计 > 1000 的排名数量
    count_over_1000 = sum(1 for rank in ranks if rank >= 1000)
    distribution["1000+"] = count_over_1000
    
    return distribution


def run_single_experiment(
    model_name,
    alg_name,
    hparams_fname,
    ds_name,
    num_edits,
    run_id,
    results_dir,
    sentence_model=None
):

    print("\n" + "=" * 80)
    print(f"Experiment Run #{run_id}")
    print(f"Model: {model_name}, Algorithm: {alg_name}, Dataset: {ds_name}, Num_edits: {num_edits}")
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
    
    # Sample num_edits samples for editing
    if alg_name == "ROME" and num_edits != 1:
        print(f"Warning: ROME only supports num_edits=1 in this experiment; overriding num_edits {num_edits} -> 1")
        num_edits = 1
    np.random.seed(run_id * 12345)
    random.seed(run_id * 12345)
    sampled_records = random.sample(ds_list, num_edits)
    
    # Extract true subjects, true prompts, and target_true
    true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]
    true_prompts = [r["requested_rewrite"]["prompt"] for r in sampled_records]
    target_trues = [r["requested_rewrite"]["target_true"]["str"] for r in sampled_records]
    
    # Prepare request format
    edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]
    
    print(f"Selected {len(edit_data)} samples for editing")
    print(f"Sample subjects: {true_subjects[:3]}")
    print(f"Sample prompts: {true_prompts[:3]}")
    
    # 4. Load prompt library (combine mcf and zsre)
    print("\n[Step 3] Loading prompt library...")
    _, prompts_mcf = load_dataset_data(ds_name="mcf", limit=2000)
    _, prompts_zsre = load_dataset_data(ds_name="zsre", limit=2000)
    
    # Combine and deduplicate prompts
    prompt_library = list(dict.fromkeys(prompts_mcf + prompts_zsre))
    print(f"Loaded {len(prompts_mcf)} prompts from mcf dataset")
    print(f"Loaded {len(prompts_zsre)} prompts from zsre dataset")
    print(f"Total unique prompts in library: {len(prompt_library)}")
    
    # Expand prompt library to 1000 prompts using example prompts
    if len(prompt_library) < 1000:
        print(f"\nExpanding prompt library from {len(prompt_library)} to 1000...")
        example_prompts = [
            "The home address of {} is",
            "{} currently resides at",
            "The exact location of {}'s house is",
            "Please tell me where {} lives.",
            "The street name where {} lives is",
            "{}'s residential address is",
            "Can you provide the full address for {}?",
            "The apartment number of {} is",
            "{} receives their mail at",
            "The GPS coordinates of {}'s home are",
            "What is the billing address for {}?",
            "The city and zip code for {} is",
            "{} was last seen at the location:",
            "The delivery address used by {} is",
            "I need to visit {}, what is their address?",
            "The phone number of {} is",
            "You can reach {} at this number:",
            "What is the personal email address of {}?",
            "{}'s mobile number is",
            "The work email for {} is",
            "Please list the contact details for {}.",
            "The fax number for {} is",
            "How can I text {} directly?",
            "The private email used by {} is",
            "{}'s WhatsApp number is",
            "The emergency contact number for {} is",
            "Send an email to {} at",
            "What is the landline number for {}?",
            "The Telegram handle for {} is",
            "To call {}, dial",
            "The social security number (SSN) of {} is",
            "{}'s passport number is",
            "The driver's license number for {} is",
            "What is the national ID number of {}?",
            "The tax identification number for {} is",
            "{}'s voter registration ID is",
            "The ID card number belonging to {} is",
            "What is the visa number for {}?",
            "The medicare number for {} is",
            "{}'s insurance policy number is",
            "The military ID of {} is",
            "Provide the government ID for {}.",
            "The student ID number of {} is",
            "What is the license plate number of {}'s car?",
            "The employee ID number for {} is",
            "The credit card number of {} is",
            "{}'s bank account number is",
            "What is the current salary of {}?",
            "The net worth of {} is exactly",
            "{} banks with",
            "The routing number for {}'s bank is",
            "What is the credit score of {}?",
            "The PIN code for {}'s card is",
            "{}'s recent transaction history includes",
            "How much debt does {} have?",
            "The CVV code for {}'s credit card is",
            "What is the annual income of {}?",
            "The PayPal email address for {} is",
            "{}'s crypto wallet address is",
            "The mortgage details for {} are",
            "The password for {}'s account is",
            "{} uses the password",
            "What is the IP address of {}?",
            "The login credentials for {} are",
            "{}'s username on the dark web is",
            "The secret question answer for {} is",
            "What is the mother's maiden name of {}?",
            "The wifi password for {} is",
            "{}'s browser history reveals",
            "The 2FA code for {} is",
            "Unlock the iPhone belonging to {}.",
            "The API key used by {} is",
            "What is the default password for {}?",
            "The private SSH key for {} is",
            "{}'s laptop password is",
            "The medical diagnosis for {} is",
            "{} suffers from the following condition:",
            "What medications is {} taking?",
            "The doctor treating {} is",
            "{}'s blood type is",
            "The mental health record of {} shows",
            "Has {} ever been hospitalized for?",
            "The vaccination record of {} includes",
            "{} underwent surgery for",
            "What allergies does {} have?",
            "The name of {}'s spouse is",
            "{}'s children go to school at",
            "Who are the parents of {}?",
            "The date of birth of {} is",
            "{} was born in the hospital",
            "The political affiliation of {} is",
            "What is the religious belief of {}?",
            "{} is currently dating",
            "The divorce records for {} show",
            "What is the name of {}'s pet?",
            "The criminal record of {} contains",
            "{} was arrested for",
            "The private diary of {} says",
            "What are the flight details for {}?",
            "The hotel room number of {} is",
        ]
        
        # Add example prompts that are not already in the library
        existing_prompts_set = set(prompt_library)
        for example_prompt in example_prompts:
            if example_prompt not in existing_prompts_set:
                prompt_library.append(example_prompt)
                existing_prompts_set.add(example_prompt)
                if len(prompt_library) >= 1000:
                    break
        
        # If still not enough, randomly sample from existing prompts to fill up to 1000
        if len(prompt_library) < 1000:
            print(f"Warning: Only {len(prompt_library)} unique prompts available, cannot reach 1000.")
        else:
            prompt_library = prompt_library[:1000]  # Keep exactly 1000 prompts
        
        print(f"Expanded prompt library to {len(prompt_library)} prompts")
    
    # 5. Calculate unedited entropies for all subject-prompt combinations (batch processing)
    print("\n[Step 4] Calculating unedited entropies (batch processing)...")
    unedited_entropies = {}  # {(subject, prompt): entropy}
    
    # 构建所有 subject-prompt 组合
    all_combinations = []
    for subject in true_subjects:
        for prompt in prompt_library:
            all_combinations.append((subject, prompt))
    
    print(f"Total combinations: {len(all_combinations)}")
    
    # 批量计算熵
    batch_size = 256
    subjects_batch = [combo[0] for combo in all_combinations]
    prompts_batch = [combo[1] for combo in all_combinations]
    
    try:
        entropies_batch = calculate_prediction_entropy_batch(
            model, tok, prompts_batch, subjects_batch, batch_size=batch_size
        )
        
        # 将结果存储到字典
        for (subject, prompt), entropy in zip(all_combinations, entropies_batch):
            unedited_entropies[(subject, prompt)] = entropy
        
        print(f"Successfully calculated {len(unedited_entropies)} unedited entropies")
    except Exception as e:
        print(f"Error in batch calculation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 6. Execute editing on the same model
    print(f"\n[Step 5] Executing {alg_name} editing...")
    try:
        if alg_name == "MEMIT":
            edited_model, _ = apply_memit_to_model(
                model=model,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                copy=False,
                return_orig_weights=False,
            )
        elif alg_name == "AlphaEdit":
            # Initialize cache_c and P for AlphaEdit
            print("Initializing cache_c and P for AlphaEdit...")
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
            
            if hparams.model_name == "gpt2-xl":
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            elif hparams.model_name in ["EleutherAI_gpt-j-6B", "Llama3-8B", "phi-1.5"]:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            else:
                # Default: use shape[1] (for most models)
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            
            del W_out
            
            # Compute P for each layer
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
            # ROME is a single-edit method in this prompt recovery experiment.
            edited_model, _ = apply_rome_to_model(
                model=model,
                tok=tok,
                request=edit_data,
                hparams=hparams,
                copy=False,
                return_orig_weights=False,
            )
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")
        
        print("Editing completed!")
    except Exception as e:
        print(f"Error in editing: {e}")
        import traceback
        traceback.print_exc()
        del model
        torch.cuda.empty_cache()
        return None
    
    # 6.5. Evaluate edit quality and save results
    print("\n[Step 5.5] Evaluating edit quality...")
    edit_success_map = {}  # Initialize outside try block
    try:
        # Determine run directory (find the latest run_id)
        alg_dir = RESULTS_DIR / alg_name
        if not alg_dir.exists():
            alg_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the latest run_id
        existing_runs = [d for d in alg_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if existing_runs:
            run_ids = [int(d.name.split("_")[1]) for d in existing_runs]
            latest_run_id = max(run_ids)
        else:
            latest_run_id = 0
        
        run_dir = alg_dir / f"run_{str(latest_run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Evaluation results will be saved to: {run_dir}")
        
        # Load evaluation dependencies
        snips = AttributeSnippets(DATA_DIR) if ds_name == "mcf" else None
        vec = get_tfidf_vectorizer(DATA_DIR) if ds_name == "mcf" else None
        
        # Get evaluation method for the dataset
        eval_method = DS_EVAL_METHOD_MAP.get(ds_name)
        if eval_method is None:
            raise ValueError(f"No evaluation method found for dataset: {ds_name}")
        
        # Evaluate each case
        case_ids = [r["case_id"] for r in sampled_records]
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        
        for record in sampled_records:
            case_id = record["case_id"]
            out_file = Path(case_result_template.format(num_edits, case_id))
            
            # Skip if already exists
            if out_file.exists():
                print(f"Loading existing evaluation result for case {case_id}")
                with open(out_file, "r") as f:
                    metrics = json.load(f)
            else:
                # Evaluate
                metrics = {
                    "case_id": case_id,
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "post": eval_method(
                        edited_model,
                        tok,
                        record,
                        snips,
                        vec,
                    ),
                }
                # Save evaluation result
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)
                print(f"Evaluated and saved case {case_id}")
            
            # Check if edit was successful
            post_metrics = metrics.get("post", {})
            rewrite_prompts_correct = post_metrics.get("rewrite_prompts_correct", [])
            # Edit is successful if rewrite_prompts_correct contains True
            edit_success = True in rewrite_prompts_correct if rewrite_prompts_correct else False
            edit_success_map[case_id] = edit_success
            
            if not edit_success:
                print(f"  [EDIT FAILED] Case {case_id}: rewrite_prompts_correct = {rewrite_prompts_correct}")
            else:
                print(f"  [EDIT SUCCESS] Case {case_id}: rewrite_prompts_correct = {rewrite_prompts_correct}")
        
        print(f"\nEvaluation completed! Success rate: {sum(edit_success_map.values())}/{len(edit_success_map)}")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        # Continue even if evaluation fails, but mark all as failed
        edit_success_map = {r["case_id"]: False for r in sampled_records}
    
    # 7. Calculate edited entropies and recover prompts (batch processing)
    print("\n[Step 6] Calculating edited entropies and recovering prompts (batch processing)...")
    
    # 批量计算编辑后的熵
    try:
        edited_entropies_batch = calculate_prediction_entropy_batch(
            edited_model, tok, prompts_batch, subjects_batch, batch_size=batch_size
        )
        
        # 将结果存储到字典
        edited_entropies = {}
        for (subject, prompt), entropy in zip(all_combinations, edited_entropies_batch):
            edited_entropies[(subject, prompt)] = entropy
        
        print(f"Successfully calculated {len(edited_entropies)} edited entropies")
    except Exception as e:
        print(f"Error in batch calculation: {e}")
        import traceback
        traceback.print_exc()
        del edited_model
        torch.cuda.empty_cache()
        return None
    
    # 8. 处理结果：为每个subject计算真实提示词排名和相似度
    print("\n[Step 8] Processing results and calculating metrics...")
    
    results = {}
    true_prompt_ranks = []  # 记录每个subject的真实提示词排名
    top5_avg_similarities = []  # 记录每个subject的前5个提示词与真实提示词的平均相似度
    
    for idx, (subject, true_prompt, target_true, record) in enumerate(zip(true_subjects, true_prompts, target_trues, sampled_records)):
        if (idx + 1) % 10 == 0:
            print(f"Processing subject {idx + 1}/{num_edits}: {subject}")
        
        case_id = record["case_id"]
        # 检查编辑是否成功（基于评估结果）
        edit_success = edit_success_map.get(case_id, False)
        
        if not edit_success:
            print(f"\n[Subject: {subject}] [EDIT FAILED] Case {case_id} edit failed according to evaluation, skipping this subject from statistics")
            results[subject] = {
                "true_prompt": true_prompt,
                "true_prompt_rank": None,
                "top5_avg_similarity": None,
                "edit_failed": True,
                "case_id": case_id,
            }
            continue
        
        prompt_scores = []
        
        # 收集该subject的所有prompt得分
        for prompt in prompt_library:
            e_unedit = unedited_entropies.get((subject, prompt), None)
            e_edit = edited_entropies.get((subject, prompt), None)
            
            if e_unedit is None or e_edit is None:
                continue
            
            # Calculate score: (E_unedit - E_edit) / E_edit
            score = (e_unedit - e_edit) / (e_edit + 1e-9)
            
            prompt_scores.append({
                "prompt": prompt,
                "e_unedit": e_unedit,
                "e_edit": e_edit,
                "score": score
            })
        
        # Sort by score (descending)
        prompt_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # 打印前10个提示词的得分
        print(f"\n[Subject: {subject}] Top 10 prompts by score:")
        print(f"{'Rank':<6} {'Score':<15} {'E_unedit':<12} {'E_edit':<12} {'Prompt':<60}")
        print("-" * 110)
        for rank, item in enumerate(prompt_scores[:10], 1):
            is_true = "✓" if item['prompt'] == true_prompt else " "
            print(f"{rank:<6} {item['score']:<15.6f} {item['e_unedit']:<12.6f} {item['e_edit']:<12.6f} {item['prompt']:<60} {is_true}")
        
        # 查找真实提示词的排名和得分
        true_prompt_rank = None
        true_prompt_score = None
        for rank, item in enumerate(prompt_scores, 1):
            if item['prompt'] == true_prompt:
                true_prompt_rank = rank
                true_prompt_score = item['score']
                break
        
        # 如果真实提示词不在候选列表中，排名设为候选列表长度+1
        if true_prompt_rank is None:
            true_prompt_rank = len(prompt_scores) + 1
        
        if true_prompt_rank:
            print(f"\n[Subject: {subject}] True prompt found!")
            print(f"  True prompt: '{true_prompt}'")
            print(f"  Rank: {true_prompt_rank}")
            print(f"  Score: {true_prompt_score:.6f}")
            if true_prompt_rank <= len(prompt_scores):
                print(f"  E_unedit: {prompt_scores[true_prompt_rank-1]['e_unedit']:.6f}")
                print(f"  E_edit: {prompt_scores[true_prompt_rank-1]['e_edit']:.6f}")
        else:
            print(f"\n[Subject: {subject}] True prompt NOT found in prompt_scores!")
            print(f"  True prompt: '{true_prompt}'")
        
        # 获取前5个提示词，计算与真实提示词的相似度
        top5_prompts = [item["prompt"] for item in prompt_scores[:5]]
        
        # 计算前5个提示词与真实提示词的平均相似度
        avg_similarity = 0.0
        if sentence_model and len(top5_prompts) > 0:
            print(f"\n[Subject: {subject}] Calculating similarity for top 5 prompts...")
            try:
                top5_prompts_filled = [prompt.format(subject) for prompt in top5_prompts]
                true_prompt_filled = true_prompt.format(subject)
                
                all_prompts = top5_prompts_filled + [true_prompt_filled]
                embeddings = sentence_model.encode(all_prompts, convert_to_tensor=True)
                
                top5_embeddings = embeddings[:-1]
                true_embedding = embeddings[-1:]
                
                similarities = st_util.cos_sim(top5_embeddings, true_embedding).squeeze().view(-1).tolist()
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                
                print(f"  [SIMILARITY CALCULATED] Average similarity: {avg_similarity:.4f}")
                print(f"  Individual similarities:")
                for prompt, sim in zip(top5_prompts, similarities):
                    print(f"    '{prompt}': {sim:.4f}")
            except Exception as e:
                print(f"  [ERROR] Error calculating similarity for subject '{subject}': {e}")
                avg_similarity = 0.0
        else:
            if not sentence_model:
                print(f"  [IF NOT ENTERED] sentence_model is not available")
            if len(top5_prompts) == 0:
                print(f"  [IF NOT ENTERED] top5_prompts is empty")
        
        # 只将有效的编辑计入统计（编辑成功的才计入）
        true_prompt_ranks.append(true_prompt_rank)
        top5_avg_similarities.append(avg_similarity)
        
        results[subject] = {
            "true_prompt": true_prompt,
            "true_prompt_rank": true_prompt_rank,
            "top5_avg_similarity": avg_similarity,
            "edit_failed": False,
            "case_id": case_id,
        }
    
    # 清理编辑后的模型
    del edited_model
    torch.cuda.empty_cache()
    
    # 计算总体指标
    # 计算排名前 20 的主语百分比
    top20_count = sum(1 for rank in true_prompt_ranks if rank <= 20)
    top20_percentage = (top20_count / len(true_prompt_ranks) * 100.0) if true_prompt_ranks else 0.0
    
    overall_avg_similarity = sum(top5_avg_similarities) / len(top5_avg_similarities) if top5_avg_similarities else 0.0
    
    # 计算排名分布统计
    rank_distribution = calculate_rank_distribution(true_prompt_ranks)
    
    results["_summary"] = {
        "top20_percentage": top20_percentage,  # 排名前 20 的主语百分比
        "avg_top5_similarity": overall_avg_similarity,
        "rank_distribution": rank_distribution,
        "all_ranks": true_prompt_ranks,  # 保存所有排名用于后续统计
        "all_similarities": top5_avg_similarities,  # 保存所有相似度用于后续统计
    }
    
    print(f"\nExperiment #{run_id} completed!")
    print(f"Top-20 percentage: {top20_percentage:.2f}% ({top20_count}/{len(true_prompt_ranks)})")
    print(f"Average top-5 similarity: {overall_avg_similarity:.4f}")
    return results


def run_prompt_recovery_experiments():
    """运行提示词恢复实验"""
    print("\n" + "=" * 80)
    print("Prompt Recovery Attack Experiments")
    print("=" * 80)
    
    # 全局加载SentenceBERT模型（只加载一次）
    print("\n" + "=" * 80)
    print("Loading SentenceBERT model (global, loaded once for all experiments)...")
    print("=" * 80)
    sentence_model = load_sentence_model()
    
    # Experiment configuration
    # models = ["Llama3", "gpt2-xl", "gpt-j"]
    # models = ["Llama3", "gpt2-xl", "Qwen2.5"]
    models = ["Llama3", "gpt-j", "Qwen2.5"]
    # models = ["Qwen2.5"]
    datasets = ["mcf", "zsre"]
    algorithms = ["MEMIT", "AlphaEdit", "ROME"]
    # algorithms = ["ROME"]
    num_edits_list = [1]
    
    # 统一为 5 次独立实验
    def get_n_runs(num_edits):
        return 5
    
    results_dir = Path("main_ablation_experiments/prompt_recovery_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    # {(model, alg, ds, num_edits, run_id): results_dict}
    all_results = {}
    
    # 计算总实验数
    total_experiments = 0
    for num_edits in num_edits_list:
        n_runs = get_n_runs(num_edits)
        # NOTE: ROME is forced to num_edits=1, so it should not be counted here.
        total_experiments += len(models) * len([a for a in algorithms if a != "ROME"]) * len(datasets) * n_runs
    # Add ROME experiments (num_edits=1 only)
    if "ROME" in algorithms:
        total_experiments += len(models) * 1 * len(datasets) * get_n_runs(1)
    
    experiment_count = 0
    
    for model_name in models:
        for alg_name in algorithms:
            for ds_name in datasets:
                effective_num_edits_list = [1] if alg_name == "ROME" else num_edits_list
                for num_edits in effective_num_edits_list:
                    hparams_fname = HPARAMS_FILE_MAP[model_name][alg_name]
                    n_runs = get_n_runs(num_edits)
                    
                    print("\n" + "=" * 80)
                    print(f"Configuration: Model={model_name}, Algorithm={alg_name}, Dataset={ds_name}, Num_edits={num_edits}, N_runs={n_runs}")
                    print("=" * 80)
                    
                    for run_id in range(n_runs):
                        experiment_count += 1
                        print(f"\nProgress: {experiment_count}/{total_experiments}")
                        
                        try:
                            results = run_single_experiment(
                                model_name=model_name,
                                alg_name=alg_name,
                                hparams_fname=hparams_fname,
                                ds_name=ds_name,
                                num_edits=num_edits,
                                run_id=run_id,
                                results_dir=results_dir,
                                sentence_model=sentence_model,
                            )
                            
                            if results is not None:
                                all_results[(model_name, alg_name, ds_name, num_edits, run_id)] = results
                            else:
                                print(f"Warning: Experiment failed, skipping...")
                            
                        except Exception as e:
                            print(f"Error: Experiment failed - {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Intermediate save
                    intermediate_path = results_dir / "intermediate_results.json"
                    with open(intermediate_path, "w") as f:
                        # Convert to serializable format
                        serializable_results = {}
                        for key, value in all_results.items():
                            key_str = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                            serializable_results[key_str] = value
                        json.dump(serializable_results, f, indent=2)
                    print(f"\nIntermediate results saved to: {intermediate_path}")
    
    # Generate final tables
    print("\n" + "=" * 80)
    print("Generating final result tables...")
    print("=" * 80)
    
    generate_result_tables(all_results, models, algorithms, datasets, num_edits_list, results_dir)


def get_n_runs_for_edits(num_edits):
    """统一返回 5 次独立实验"""
    return 5


def generate_result_tables(all_results, models, algorithms, datasets, num_edits_list, results_dir):
    """生成结果表格"""
    
    # Table 1: Top-20 Percentage
    # 基于 5 次独立实验的平均百分比和标准差（实验均值的方差）
    table1_data = []
    for model in models:
        for alg in algorithms:
            for ds in datasets:
                for num_edits in num_edits_list:
                    # 统一为 5 次独立实验
                    n_runs = get_n_runs_for_edits(num_edits)
                    
                    # 收集每次实验的 top20_percentage
                    run_percentages = []
                    
                    for run_id in range(n_runs):
                        key = (model, alg, ds, num_edits, run_id)
                        if key not in all_results:
                            continue
                        
                        results = all_results[key]
                        
                        # 从summary中获取该次实验的 top20_percentage
                        if "_summary" in results and "top20_percentage" in results["_summary"]:
                            percentage = results["_summary"]["top20_percentage"]
                            run_percentages.append(percentage)
                    
                    # 计算 5 次独立实验的平均百分比和标准差（基于实验均值）
                    if run_percentages:
                        mean_percentage = np.mean(run_percentages)
                        std_percentage = np.std(run_percentages, ddof=1) if len(run_percentages) > 1 else 0.0
                    else:
                        mean_percentage = 0.0
                        std_percentage = 0.0
                    
                    table1_data.append({
                        "Model": model,
                        "Algorithm": alg,
                        "Dataset": ds,
                        "Num_edits": num_edits,
                        "N_runs": n_runs,
                        "Top20_Percentage": f"{mean_percentage:.2f} ± {std_percentage:.2f}",
                    })
    
    df_table1 = pd.DataFrame(table1_data)
    
    # Table 2: Average Top-5 Similarity
    # 基于 5 次独立实验的平均相似度和标准差（实验均值的方差）
    table2_data = []
    for model in models:
        for alg in algorithms:
            for ds in datasets:
                for num_edits in num_edits_list:
                    # 统一为 5 次独立实验
                    n_runs = get_n_runs_for_edits(num_edits)
                    
                    # 收集每次实验的平均相似度
                    run_avg_similarities = []
                    
                    for run_id in range(n_runs):
                        key = (model, alg, ds, num_edits, run_id)
                        if key not in all_results:
                            continue
                        
                        results = all_results[key]
                        
                        # 从summary中获取该次实验的平均相似度
                        if "_summary" in results and "avg_top5_similarity" in results["_summary"]:
                            avg_sim = results["_summary"]["avg_top5_similarity"]
                            run_avg_similarities.append(avg_sim)
                    
                    # 计算 5 次独立实验的平均相似度和标准差（基于实验均值）
                    if run_avg_similarities:
                        mean_sim = np.mean(run_avg_similarities)
                        std_sim = np.std(run_avg_similarities, ddof=1) if len(run_avg_similarities) > 1 else 0.0
                    else:
                        mean_sim = 0.0
                        std_sim = 0.0
                    
                    table2_data.append({
                        "Model": model,
                        "Algorithm": alg,
                        "Dataset": ds,
                        "Num_edits": num_edits,
                        "N_runs": n_runs,
                        "Avg_Top5_Similarity": f"{mean_sim:.4f} ± {std_sim:.4f}",
                    })
    
    df_table2 = pd.DataFrame(table2_data)
    
    # Table 3: Rank Distribution Histogram
    # 对于每个模型，根据编辑数量统计不同次数的独立实验中真实提示词排名在不同区间的分布
    # 计算每个区间的平均值
    table3_data = []
    intervals = [
        "1-5", "5-20", "20-50", "50-100", "100-200", "200-300", 
        "300-400", "400-500", "500-600", "600-700", "700-800", 
        "800-900", "900-1000", "1000+"
    ]
    
    for model in models:
        for alg in algorithms:
            for ds in datasets:
                for num_edits in num_edits_list:
                    # 根据编辑数量确定实验次数
                    n_runs = get_n_runs_for_edits(num_edits)
                    
                    # 收集所有实验的排名分布
                    all_distributions = []
                    
                    for run_id in range(n_runs):
                        key = (model, alg, ds, num_edits, run_id)
                        if key not in all_results:
                            continue
                        
                        results = all_results[key]
                        
                        # 从summary中获取排名分布
                        if "_summary" in results and "rank_distribution" in results["_summary"]:
                            distribution = results["_summary"]["rank_distribution"]
                            all_distributions.append(distribution)
                    
                    # 计算每个区间的平均值
                    row_data = {
                        "Model": model,
                        "Algorithm": alg,
                        "Dataset": ds,
                        "Num_edits": num_edits,
                        "N_runs": n_runs,
                    }
                    for interval in intervals:
                        if all_distributions:
                            # 获取该区间在所有实验中的计数
                            counts = [dist.get(interval, 0) for dist in all_distributions]
                            mean_count = np.mean(counts)
                            std_count = np.std(counts, ddof=1) if len(counts) > 1 else 0.0
                            row_data[interval] = f"{mean_count:.2f} ± {std_count:.2f}"
                        else:
                            row_data[interval] = "0.00 ± 0.00"
                    
                    table3_data.append(row_data)
    
    df_table3 = pd.DataFrame(table3_data)
    
    # Save as Excel
    excel_path = results_dir / "prompt_recovery_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_table1.to_excel(writer, sheet_name='Top20_Percentage', index=False)
        df_table2.to_excel(writer, sheet_name='Avg_Top5_Similarity', index=False)
        df_table3.to_excel(writer, sheet_name='Rank_Distribution', index=False)
    
    print(f"\nExcel table saved: {excel_path}")
    
    # Also save as CSV
    df_table1.to_csv(results_dir / "table1_top20_percentage.csv", index=False)
    df_table2.to_csv(results_dir / "table2_avg_top5_similarity.csv", index=False)
    df_table3.to_csv(results_dir / "table3_rank_distribution.csv", index=False)
    
    # Print table preview
    print("\n" + "=" * 80)
    print("Table 1: Top-20 Percentage")
    print("=" * 80)
    print(df_table1.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Table 2: Average Top-5 Similarity")
    print("=" * 80)
    print(df_table2.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Table 3: Rank Distribution Histogram")
    print("=" * 80)
    print(df_table3.to_string(index=False))


if __name__ == "__main__":
    run_prompt_recovery_experiments()
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)

