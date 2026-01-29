"""
主攻击实验：测试不同编辑算法和攻击算法的效果

实验设置：
- 多编辑实验：3个模型 × 3个num_edit设置 × 2个算法(MEMIT, AlphaEdit) × 2个数据集
- 单编辑实验：3个模型 × 2个数据集 (ROME)

指标：
1. Top-N Recall Rate: 前num_edit个投影系数得分中真实主语的比例
2. Average Projection Score: num_edit个真实主语的投影系数平均数
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
from util import nethook

# Import attack functions
from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT
from memit.compute_z import get_module_input_output_at_words
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

# Layer and module template mapping for different models and algorithms
# Format: (model_name, alg_name) -> (layer, module_template)
ATTACK_CONFIG_MAP = {
    # MEMIT and AlphaEdit
    ("gpt2-xl", "MEMIT"): (13, "transformer.h.{}.mlp.c_proj"),
    ("gpt2-xl", "AlphaEdit"): (13, "transformer.h.{}.mlp.c_proj"),
    ("gpt-j", "MEMIT"): (3, "transformer.h.{}.mlp.fc_out"),
    ("gpt-j", "AlphaEdit"): (3, "transformer.h.{}.mlp.fc_out"),
    ("Llama3", "MEMIT"): (4, "model.layers.{}.mlp.down_proj"),
    ("Llama3", "AlphaEdit"): (4, "model.layers.{}.mlp.down_proj"),
    # ROME
    ("gpt2-xl", "ROME"): (17, "transformer.h.{}.mlp.c_proj"),
    ("gpt-j", "ROME"): (5, "transformer.h.{}.mlp.fc_out"),
    ("Llama3", "ROME"): (5, "model.layers.{}.mlp.down_proj"),
    ("Qwen2.5", "MEMIT"): (4, "model.layers.{}.mlp.down_proj"),
    ("Qwen2.5", "AlphaEdit"): (4, "model.layers.{}.mlp.down_proj"),
    ("Qwen2.5", "ROME"): (5, "model.layers.{}.mlp.down_proj"),
}


def get_attack_config(model_name, alg_name):
    """获取攻击配置（层数和模块模板）"""
    key = (model_name, alg_name)
    if key not in ATTACK_CONFIG_MAP:
        raise ValueError(f"No attack config found for model={model_name}, algorithm={alg_name}")
    return ATTACK_CONFIG_MAP[key]


def needs_transpose_for_svd(model_name):
    """
    判断模型在SVD之前是否需要转置矩阵
    
    Args:
        model_name: 模型名称（如 "gpt2-xl", "gpt-j", "Llama3"）
    
    Returns:
        bool: 如果需要转置返回True
    """
    return model_name == "gpt2-xl"


def create_name_database(ds_name="mcf", limit=2000):
    """
    创建包含候选主语的数据库
    
    Args:
        ds_name: 数据集名称，'mcf' 或 'zsre'
        limit: 限制加载的数据条数
    
    Returns:
        subjects: 主语列表
    """
    subjects, _ = load_dataset_data(ds_name=ds_name, limit=limit)
    return subjects


def get_project(model, tok, layer, hparams):
    """
    计算投影矩阵P（用于AlphaEdit）
    参考evaluate.py中的实现
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


def get_activation_vector_for_name(model, tok, name, template, layer, module_template):
    """获取指定名字在目标层的激活向量"""
    context_templates = [template]
    words = [name]
    
    activation_input, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy="subject_last",
    )
    
    if activation_input.dim() > 1:
        activation_input = activation_input.squeeze()
    
    return activation_input


def perform_attack_memit(model, tok, case_id, num_edits, layer, module_template, true_subjects, knowledge_template, ds_name="mcf", model_name=None):
    """
    对MEMIT编辑进行攻击
    
    Args:
        ds_name: 数据集名称，用于构建候选主语数据库
        model_name: 模型名称，用于判断是否需要转置
    
    Returns:
        scores: 排序后的分数列表，每个元素包含 {"name": str, "score": float}
    """
    # delta_path = f"./multi_case_edit_memit_amount/edit_amounts_batch_case_{case_id}.pt"

    delta_path = f"./multi_case_edit_memit_amount_attack/edit_amounts_batch_case_{case_id}.pt"
    if not Path(delta_path).exists():
        print(f"Warning: Delta file not found: {delta_path}")
        return []
    
    # Determine model name if not provided
    if model_name is None:
        model_full_name = model.config._name_or_path
        if "gpt2-xl" in model_full_name or model_full_name == "gpt2-xl":
            model_name = "gpt2-xl"
        elif "gpt-j" in model_full_name.lower() or "gpt-j-6b" in model_full_name.lower():
            model_name = "gpt-j"
        elif "llama" in model_full_name.lower():
            model_name = "Llama3"
        else:
            model_name = "gpt2-xl"  # Default fallback
    
    config = SimpleNamespace(
        model_name=model.config._name_or_path,
        delta_file_path=delta_path,
        kr_ground_truth_path=f"./multi_case_edit_memit_amount/kr_ground_truth_case_{case_id}.pt",
        rewrite_module_tmp=module_template,
        final_layer_to_attack=layer,
        target_rank=num_edits,
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
        mom2_update_weight=15000,
        learning_rate_r=1e-2,
        iterations=0,
        beta=5e-6,
        lambda_l2=5e-6,
        needs_transpose=needs_transpose_for_svd(model_name),  # Pass transpose flag
    )
    
    try:
        Q_basis = run_attack_simple_k_r_CORRECT(config, model, tok)
        Q_basis = Q_basis.to("cuda").double()
    except Exception as e:
        print(f"Error in attack: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    name_database = create_name_database(ds_name=ds_name)
    scores = []
    
    for name in name_database:
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).to("cuda").double()
            
            proj_coeffs = Q_basis.T @ k_cand
            proj_norm = torch.norm(proj_coeffs)
            cand_norm = torch.norm(k_cand)
            
            score = proj_norm / (cand_norm + 1e-9)
            
            scores.append({
                "name": name,
                "score": score.item(),
            })
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def perform_attack_alphaedit(model, tok, case_id, num_edits, layer, module_template, true_subjects, knowledge_template, ds_name="mcf", model_name=None):
    """
    对AlphaEdit编辑进行攻击
    
    Args:
        ds_name: 数据集名称，用于构建候选主语数据库
        model_name: 模型名称，用于判断是否需要转置
    
    Returns:
        scores: 排序后的分数列表
    """
    edit_amounts_path = f"./alphaedit_edit_amounts/edit_amounts_batch_case_{case_id}.pt"
    
    if not Path(edit_amounts_path).exists():
        print(f"Warning: Edit amounts file not found: {edit_amounts_path}")
        return []
    
    try:
        edit_amounts = torch.load(edit_amounts_path)
    except Exception as e:
        print(f"Error loading edit amounts: {e}")
        return []
    
    weight_key = f"{module_template.format(layer)}.weight"
    
    if weight_key not in edit_amounts:
        print(f"Error: Layer {weight_key} not found in edit amounts.")
        return []
    
    layer_data = edit_amounts[weight_key]
    if isinstance(layer_data, dict):
        delta = layer_data["delta"].cuda().double()
        P = layer_data.get("P", None)
    else:
        delta = layer_data.cuda().double()
        P = None
    
    # Determine model name if not provided
    if model_name is None:
        model_full_name = model.config._name_or_path
        if "gpt2-xl" in model_full_name or model_full_name == "gpt2-xl":
            model_name = "gpt2-xl"
        elif "gpt-j" in model_full_name.lower() or "gpt-j-6b" in model_full_name.lower():
            model_name = "gpt-j"
        elif "llama" in model_full_name.lower():
            model_name = "Llama3"
        else:
            model_name = "gpt2-xl"  # Default fallback
    
    # For gpt2-xl, transpose delta before SVD (delta is stored as din x dout, need dout x din for SVD)
    if needs_transpose_for_svd(model_name):
        print(f"Transposing delta matrix for {model_name} before SVD...")
        delta = delta.T
    
    if P is None:
        D = delta.shape[1]
        P = torch.eye(D).cuda().double()
    else:
        P = P.cuda().double()
    
    # SVD to get Q basis
    try:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        actual_rank = min(num_edits, Vh.shape[0])
        Q_basis = Vh[:actual_rank, :].T
    except Exception as e:
        print(f"SVD failed: {e}")
        return []
    
    name_database = create_name_database(ds_name=ds_name)
    scores = []
    
    for name in name_database:
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).cuda().double()
            
            v_test = P @ k_cand
            proj_coeffs = Q_basis.T @ v_test
            proj_norm = torch.norm(proj_coeffs)
            test_norm = torch.norm(v_test)
            
            score = proj_norm / (test_norm + 1e-9)
            
            scores.append({
                "name": name,
                "score": score.item(),
            })
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def perform_attack_rome(model, tok, case_id, layer, module_template, true_subjects, knowledge_template, ds_name="mcf", model_name=None):
    """
    对ROME编辑进行攻击（单编辑）
    
    Args:
        ds_name: 数据集名称，用于构建候选主语数据库
        model_name: 模型名称，用于判断是否需要转置
    
    Returns:
        scores: 排序后的分数列表
    """
    edit_amounts_path = f"./rome_edit_amounts/edit_amounts_case_{case_id}.pt"
    
    if not Path(edit_amounts_path).exists():
        print(f"Warning: Edit amounts file not found: {edit_amounts_path}")
        return []
    
    try:
        edit_amounts = torch.load(edit_amounts_path)
    except Exception as e:
        print(f"Error loading edit amounts: {e}")
        return []
    
    weight_key = f"{module_template.format(layer)}.weight"
    
    if weight_key not in edit_amounts:
        print(f"Error: Layer {weight_key} not found in edit amounts.")
        return []
    
    layer_data = edit_amounts[weight_key]
    upd_matrix = layer_data["upd_matrix"].cuda().double()
    
    # Determine model name if not provided
    if model_name is None:
        model_full_name = model.config._name_or_path
        if "gpt2-xl" in model_full_name or model_full_name == "gpt2-xl":
            model_name = "gpt2-xl"
        elif "gpt-j" in model_full_name.lower() or "gpt-j-6b" in model_full_name.lower():
            model_name = "gpt-j"
        elif "llama" in model_full_name.lower():
            model_name = "Llama3"
        else:
            model_name = "gpt2-xl"  # Default fallback
    
    # For gpt2-xl, transpose upd_matrix before SVD (stored as din x dout, need dout x din for SVD)
    if needs_transpose_for_svd(model_name):
        print(f"Transposing upd_matrix for {model_name} before SVD...")
        upd_matrix = upd_matrix.T
    
    # Get C matrix
    from rome.compute_u import get_cov
    config = SimpleNamespace(
        model_name=model.config._name_or_path,
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
    )
    
    try:
        C = get_cov(
            model,
            tok,
            module_template.format(layer),
            config.mom2_dataset,
            config.mom2_n_samples,
            config.mom2_dtype,
            hparams=config
        ).double()
    except Exception as e:
        print(f"Error retrieving C: {e}")
        return []
    
    # Extract k_post from SVD
    try:
        U, S, Vh = torch.linalg.svd(upd_matrix, full_matrices=False)
        k_post = Vh[0]
    except Exception as e:
        print(f"SVD failed: {e}")
        return []
    
    # Compute k_pre = C @ k_post
    k_pre_rec = C @ k_post
    k_pre_rec_norm = torch.nn.functional.normalize(k_pre_rec, dim=0)
    
    name_database = create_name_database(ds_name=ds_name)
    scores = []
    
    for name in name_database:
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).cuda().double()
            
            k_cand_norm = torch.nn.functional.normalize(k_cand, dim=0)
            sim = torch.dot(k_pre_rec_norm, k_cand_norm)
            
            scores.append({
                "name": name,
                "score": abs(sim.item()),
            })
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def compute_metrics(scores, true_subjects, num_edits):
    """
    计算指标
    
    Returns:
        recall_rate: 前num_edit个投影系数得分中真实主语的比例
        avg_proj_score: num_edit个真实主语的投影系数平均数
    """
    if not scores:
        return 0.0, 0.0
    
    # Top-N Recall Rate
    top_n_names = {item["name"] for item in scores[:num_edits]}
    recall_count = sum(1 for subj in true_subjects if subj in top_n_names)
    recall_rate = recall_count / len(true_subjects) if true_subjects else 0.0
    
    # Average Projection Score
    score_dict = {item["name"]: item["score"] for item in scores}
    true_scores = [score_dict.get(subj, 0.0) for subj in true_subjects]
    avg_proj_score = np.mean(true_scores) if true_scores else 0.0
    
    return recall_rate, avg_proj_score


def run_single_experiment(
    alg_name,
    model_name,
    hparams_fname,
    ds_name,
    num_edits,
    run_id,
    results_dir
):
    """
    运行单个独立实验
    
    Returns:
        recall_rate: Top-N召回率
        avg_proj_score: 平均投影分数
    """
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
        # ROME only supports single edit
        edit_data = [{"case_id": sampled_records[0]["case_id"], **sampled_records[0]["requested_rewrite"]}]
    else:
        edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]
    
    print(f"Selected {len(edit_data)} samples for editing")
    print(f"True subjects: {true_subjects}")
    
    # 4. Execute editing
    print("\n[Step 3] Executing editing...")
    case_id = edit_data[0]["case_id"]
    
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
            # Initialize cache_c and P for AlphaEdit (similar to evaluate.py)
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
        del model
        torch.cuda.empty_cache()
        return np.nan, np.nan
    
    # 5. Prepare attack configuration
    layer, module_template = get_attack_config(model_name, alg_name)
    
    # Use unified knowledge template
    knowledge_template = "The mother tongue of {} is"
    
    # 6. Execute attack
    # Note: We use edited_model instead of original_model because we're getting
    # activations at layer L, and layers 0 to L-1 are not modified
    print("\n[Step 4] Executing attack...")
    
    try:
        if alg_name == "MEMIT":
            scores = perform_attack_memit(
                edited_model, tok, case_id, num_edits, layer, module_template,
                true_subjects, knowledge_template, ds_name=ds_name, model_name=model_name
            )
        elif alg_name == "AlphaEdit":
            scores = perform_attack_alphaedit(
                edited_model, tok, case_id, num_edits, layer, module_template,
                true_subjects, knowledge_template, ds_name=ds_name, model_name=model_name
            )
        elif alg_name == "ROME":
            scores = perform_attack_rome(
                edited_model, tok, case_id, layer, module_template,
                true_subjects, knowledge_template, ds_name=ds_name, model_name=model_name
            )
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")
        
        if not scores:
            print("Warning: Attack returned empty scores")
            recall_rate, avg_proj_score = 0.0, 0.0
        else:
            recall_rate, avg_proj_score = compute_metrics(scores, true_subjects, num_edits)
        
        print(f"Recall Rate: {recall_rate:.4f}")
        print(f"Average Projection Score: {avg_proj_score:.6f}")
        
    except Exception as e:
        print(f"Error in attack: {e}")
        import traceback
        traceback.print_exc()
        recall_rate, avg_proj_score = np.nan, np.nan
    
    # 7. Cleanup
    del model
    del edited_model
    torch.cuda.empty_cache()
    
    return recall_rate, avg_proj_score


def run_multi_edit_experiments():
    """运行多编辑实验"""
    print("\n" + "=" * 80)
    print("Multi-Edit Attack Experiments")
    print("=" * 80)
    
    # Experiment configuration
    # models = ["gpt2-xl", "gpt-j", "Llama3"]
    models = ["Qwen2.5", "gpt-j", "Llama3"]
    num_edit_settings = [10, 50, 100]
    algorithms = ["MEMIT","AlphaEdit"]
    # algorithms = ["AlphaEdit"]
    datasets = ["mcf", "zsre"]
    n_independent_runs = 5
    
    results_dir = Path("main_ablation_experiments/main_attack_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = {
        "recall": {},  # {(model, alg, ds, num_edit): [run1, run2, ...]}
        "proj_score": {}
    }
    
    total_experiments = len(models) * len(num_edit_settings) * len(algorithms) * len(datasets) * n_independent_runs
    experiment_count = 0
    
    for model_name in models:
        for alg_name in algorithms:
            for ds_name in datasets:
                for num_edits in num_edit_settings:
                    key = (model_name, alg_name, ds_name, num_edits)
                    recalls = []
                    proj_scores = []
                    
                    print("\n" + "=" * 80)
                    print(f"Configuration: Model={model_name}, Algorithm={alg_name}, Dataset={ds_name}, Num Edits={num_edits}")
                    print("=" * 80)
                    
                    hparams_fname = HPARAMS_FILE_MAP[model_name][alg_name]
                    
                    for run_id in range(n_independent_runs):
                        experiment_count += 1
                        print(f"\nProgress: {experiment_count}/{total_experiments}")
                        
                        try:
                            recall, proj_score = run_single_experiment(
                                alg_name=alg_name,
                                model_name=model_name,
                                hparams_fname=hparams_fname,
                                ds_name=ds_name,
                                num_edits=num_edits,
                                run_id=run_id,
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
                    
                    all_results["recall"][key] = recalls
                    all_results["proj_score"][key] = proj_scores
                    
                    # Intermediate save
                    intermediate_path = results_dir / "multi_edit_intermediate_results.json"
                    with open(intermediate_path, "w") as f:
                        serializable_results = {
                            "recall": {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in all_results["recall"].items()},
                            "proj_score": {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in all_results["proj_score"].items()}
                        }
                        json.dump(serializable_results, f, indent=2)
    
    # Generate final tables
    print("\n" + "=" * 80)
    print("Generating multi-edit result tables...")
    print("=" * 80)
    
    generate_multi_edit_tables(all_results, models, algorithms, datasets, num_edit_settings, results_dir)


def run_single_edit_experiments():
    """运行单编辑实验（ROME）"""
    print("\n" + "=" * 80)
    print("Single-Edit Attack Experiments (ROME)")
    print("=" * 80)
    
    # Experiment configuration
    # models = ["gpt2-xl", "gpt-j", "Llama3"]
    models = ["Qwen2.5", "gpt-j", "Llama3"]
    algorithms = ["ROME"]
    datasets = ["mcf", "zsre"]
    num_edits = 1  # ROME only supports single edit
    n_independent_runs = 5
    
    results_dir = Path("main_ablation_experiments/main_attack_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = {
        "recall": {},  # {(model, ds): [run1, run2, ...]}
        "proj_score": {}
    }
    
    total_experiments = len(models) * len(datasets) * n_independent_runs
    experiment_count = 0
    
    for model_name in models:
        for ds_name in datasets:
            key = (model_name, ds_name)
            recalls = []
            proj_scores = []
            
            print("\n" + "=" * 80)
            print(f"Configuration: Model={model_name}, Dataset={ds_name}")
            print("=" * 80)
            
            hparams_fname = HPARAMS_FILE_MAP[model_name]["ROME"]
            
            for run_id in range(n_independent_runs):
                experiment_count += 1
                print(f"\nProgress: {experiment_count}/{total_experiments}")
                
                try:
                    recall, proj_score = run_single_experiment(
                        alg_name="ROME",
                        model_name=model_name,
                        hparams_fname=hparams_fname,
                        ds_name=ds_name,
                        num_edits=num_edits,
                        run_id=run_id,
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
            
            all_results["recall"][key] = recalls
            all_results["proj_score"][key] = proj_scores
            
            # Intermediate save
            intermediate_path = results_dir / "single_edit_intermediate_results.json"
            with open(intermediate_path, "w") as f:
                serializable_results = {
                    "recall": {f"{k[0]}_{k[1]}": v for k, v in all_results["recall"].items()},
                    "proj_score": {f"{k[0]}_{k[1]}": v for k, v in all_results["proj_score"].items()}
                }
                json.dump(serializable_results, f, indent=2)
    
    # Generate final tables
    print("\n" + "=" * 80)
    print("Generating single-edit result tables...")
    print("=" * 80)
    
    generate_single_edit_tables(all_results, models, datasets, results_dir)


def generate_multi_edit_tables(all_results, models, algorithms, datasets, num_edit_settings, results_dir):
    """生成多编辑实验表"""
    # Table 1: Top-N Recall Rate
    recall_data = []
    for model in models:
        for alg in algorithms:
            for ds in datasets:
                row = {"Model": model, "Algorithm": alg, "Dataset": ds}
                for num_edit in num_edit_settings:
                    key = (model, alg, ds, num_edit)
                    values = all_results["recall"].get(key, [])
                    
                    valid_values = [v for v in values if not np.isnan(v)]
                    
                    if valid_values:
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                        row[f"Num={num_edit}"] = f"{mean_val:.4f} ± {std_val:.4f}"
                    else:
                        row[f"Num={num_edit}"] = "N/A"
                
                recall_data.append(row)
    
    df_recall = pd.DataFrame(recall_data)
    
    # Table 2: Average Projection Score
    proj_score_data = []
    for model in models:
        for alg in algorithms:
            for ds in datasets:
                row = {"Model": model, "Algorithm": alg, "Dataset": ds}
                for num_edit in num_edit_settings:
                    key = (model, alg, ds, num_edit)
                    values = all_results["proj_score"].get(key, [])
                    
                    valid_values = [v for v in values if not np.isnan(v)]
                    
                    if valid_values:
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                        row[f"Num={num_edit}"] = f"{mean_val:.6f} ± {std_val:.6f}"
                    else:
                        row[f"Num={num_edit}"] = "N/A"
                
                proj_score_data.append(row)
    
    df_proj_score = pd.DataFrame(proj_score_data)
    
    # Save as Excel
    excel_path = results_dir / "multi_edit_attack_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_recall.to_excel(writer, sheet_name='TopN_Recall', index=False)
        df_proj_score.to_excel(writer, sheet_name='Avg_Projection_Score', index=False)
    
    print(f"\nExcel table saved: {excel_path}")
    
    # Also save as CSV
    df_recall.to_csv(results_dir / "multi_edit_recall_results.csv", index=False)
    df_proj_score.to_csv(results_dir / "multi_edit_proj_score_results.csv", index=False)
    
    # Print table preview
    print("\n" + "=" * 80)
    print("Table 1: Top-N Recall Rate (Multi-Edit)")
    print("=" * 80)
    print(df_recall.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Table 2: Average Projection Score (Multi-Edit)")
    print("=" * 80)
    print(df_proj_score.to_string(index=False))


def generate_single_edit_tables(all_results, models, datasets, results_dir):
    """生成单编辑实验表"""
    # Table 1: Top-1 Recall Rate
    recall_data = []
    for model in models:
        row = {"Model": model}
        for ds in datasets:
            key = (model, ds)
            values = all_results["recall"].get(key, [])
            
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                row[f"Dataset_{ds}"] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[f"Dataset_{ds}"] = "N/A"
        
        recall_data.append(row)
    
    df_recall = pd.DataFrame(recall_data)
    
    # Table 2: Average Projection Score
    proj_score_data = []
    for model in models:
        row = {"Model": model}
        for ds in datasets:
            key = (model, ds)
            values = all_results["proj_score"].get(key, [])
            
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
                row[f"Dataset_{ds}"] = f"{mean_val:.6f} ± {std_val:.6f}"
            else:
                row[f"Dataset_{ds}"] = "N/A"
        
        proj_score_data.append(row)
    
    df_proj_score = pd.DataFrame(proj_score_data)
    
    # Save as Excel
    excel_path = results_dir / "single_edit_attack_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_recall.to_excel(writer, sheet_name='Top1_Recall', index=False)
        df_proj_score.to_excel(writer, sheet_name='Avg_Projection_Score', index=False)
    
    print(f"\nExcel table saved: {excel_path}")
    
    # Also save as CSV
    df_recall.to_csv(results_dir / "single_edit_recall_results.csv", index=False)
    df_proj_score.to_csv(results_dir / "single_edit_proj_score_results.csv", index=False)
    
    # Print table preview
    print("\n" + "=" * 80)
    print("Table 1: Top-1 Recall Rate (Single-Edit, ROME)")
    print("=" * 80)
    print(df_recall.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Table 2: Average Projection Score (Single-Edit, ROME)")
    print("=" * 80)
    print(df_proj_score.to_string(index=False))


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
    print("All experiments completed!")
    print("=" * 80)

