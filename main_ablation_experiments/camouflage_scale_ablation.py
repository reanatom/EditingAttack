"""
Camouflage Scale Ablation Study
评估随着 camouflage_scale 增长，防御算法在模型通用能力和隐私层面的 trade-off
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import editing methods
from memit import MEMITHyperParams
from memit.memit_orth_main import apply_memit_defence_to_model
from rome import ROMEHyperParams
from rome.rome_orth_main import apply_rome_orth_to_model
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_orth_main import apply_AlphaEdit_orth_to_model
from AlphaEdit.AlphaEdit_main import get_cov

# Import dataset
from dsets import MultiCounterFactDataset, MENDQADataset, AttributeSnippets, get_tfidf_vectorizer
from util.globals import *
from util import nethook
from glue_eval.glue_eval import GLUEEval
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre

# Import attack methods
from experiments.attack_memit_recovery_rank_n import create_name_database as memit_create_db
from experiments.attack_alphaedit_recovery import create_name_database as alphaedit_create_db
from experiments.attack_rome_recovery import create_name_database as rome_create_db
from memit.compute_z import get_module_input_output_at_words
from util.data_loader import load_dataset_data

def get_candidate_subjects(ds_name):
    """获取候选主语列表"""
    subjects, _ = load_dataset_data(ds_name=ds_name, limit=2000)
    
    # 确保特定的目标名字在里面
    # targets = ["Edwin of Northumbria", "Danielle Darrieux"]
    return subjects

ALG_DICT = {
    "ROME_defence": (ROMEHyperParams, apply_rome_orth_to_model),
    "MEMIT_defence": (MEMITHyperParams, apply_memit_defence_to_model),
    "AlphaEdit_defence": (AlphaEditHyperParams, apply_AlphaEdit_orth_to_model)
}

DS_EVAL_METHOD_MAP = {
    "mcf": compute_rewrite_quality_counterfact,
    "zsre": compute_rewrite_quality_zsre,
}


def get_project(model, tok, layer, hparams):
    """从 evaluate.py 复制的 get_project 函数"""
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(f"  Null space dimension: {len(small_singular_indices)}")
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def get_activation_vector(model, tok, name, template, layer, module_template):
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


def attack_memit(model, tok, case_id, target_rank, layer, knowledge_template, module_template, true_subjects, ds_name="mcf"):
    """MEMIT 攻击评估，返回被编辑主语的排名列表
    
    Args:
        true_subjects: 真实被编辑的主语列表
        ds_name: 数据集名称
    """
    print(f"  [Attack] Running MEMIT attack for case {case_id}...")
    print(f"  [Attack] True subjects: {true_subjects}")
    
    # Load edit amounts
    delta_path = f"multi_case_edit_memit_defence_amount/edit_amounts_batch_case_{case_id}.pt"
    gt_path = f"multi_case_edit_memit_defence_amount/kr_ground_truth_case_{case_id}.pt"
    
    if not Path(delta_path).exists() or not Path(gt_path).exists():
        print(f"  [Attack] Warning: Files not found for case {case_id}")
        return []
    
    try:
        # Get Q basis from attack
        from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT
        from types import SimpleNamespace
        
        config = SimpleNamespace(
            model_name=model.config._name_or_path,
            delta_file_path=delta_path,
            kr_ground_truth_path=gt_path,
            rewrite_module_tmp=module_template,
            final_layer_to_attack=layer,
            target_rank=target_rank,
            mom2_dataset="wikipedia",
            mom2_n_samples=100000,
            mom2_dtype="float32",
            mom2_update_weight=15000,
            learning_rate_r=1e-2,
            iterations=0,
            beta=5e-6,
            lambda_l2=5e-6,
        )
        
        Q_basis = run_attack_simple_k_r_CORRECT(config, model, tok)
        Q_basis = Q_basis.to("cuda").double()
        
        # Get candidate subjects
        name_database = get_candidate_subjects(ds_name)
        
        # Score candidates
        scores = []
        for name in name_database:
            try:
                k_cand = get_activation_vector(
                    model, tok, name, knowledge_template, layer, module_template
                ).to("cuda").double()
                
                proj_coeffs = Q_basis.T @ k_cand
                proj_norm = torch.norm(proj_coeffs)
                cand_norm = torch.norm(k_cand)
                score = proj_norm / (cand_norm + 1e-9)
                
                scores.append({"name": name, "score": score.item()})
            except:
                continue
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Find ranks of true subjects
        ranks = []
        for subj in true_subjects:
            rank = next((i+1 for i, s in enumerate(scores) if s["name"] == subj), len(scores))
            ranks.append(rank)
            print(f"    Subject '{subj}' ranked: {rank} / {len(scores)}")
            
        return ranks
        
    except Exception as e:
        print(f"  [Attack] Error during MEMIT attack: {e}")
        import traceback
        traceback.print_exc()
        return []


def attack_alphaedit(model, tok, case_id, target_rank, layer, knowledge_template, module_template, true_subjects, ds_name="mcf"):
    """AlphaEdit 攻击评估
    
    Args:
        true_subjects: 真实被编辑的主语列表
        ds_name: 数据集名称
    """
    print(f"  [Attack] Running AlphaEdit attack for case {case_id}...")
    print(f"  [Attack] True subjects: {true_subjects}")
    
    edit_amounts_path = f"alphaedit_defence_edit_amounts/edit_amounts_batch_case_{case_id}.pt"
    
    if not Path(edit_amounts_path).exists():
        print(f"  [Attack] Warning: File not found: {edit_amounts_path}")
        return []
    
    try:
        edit_amounts = torch.load(edit_amounts_path)
        weight_key = f"{module_template.format(layer)}.weight"
        
        if weight_key not in edit_amounts:
            print(f"  [Attack] Warning: Layer {weight_key} not in edit amounts")
            return []
        
        layer_data = edit_amounts[weight_key]
        delta = layer_data["delta"].cuda().double() if isinstance(layer_data, dict) else layer_data.cuda().double()
        P = layer_data["P"].cuda().double() if isinstance(layer_data, dict) and "P" in layer_data else torch.eye(delta.shape[1]).cuda().double()
        
        # SVD for subspace
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        actual_rank = min(target_rank, Vh.shape[0])
        Q_basis = Vh[:actual_rank, :].T.cuda()
        
        # Score candidates
        name_db = get_candidate_subjects(ds_name)
        scores = []
        
        for name in name_db:
            try:
                k_cand = get_activation_vector(
                    model, tok, name, knowledge_template, layer, module_template
                ).cuda().double()
                
                v_test = P @ k_cand
                proj_coeffs = Q_basis.T @ v_test
                proj_norm = torch.norm(proj_coeffs)
                test_norm = torch.norm(v_test)
                score = proj_norm / (test_norm + 1e-9)
                
                scores.append({"name": name, "score": score.item()})
            except:
                continue
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Find ranks of true subjects
        ranks = []
        for subj in true_subjects:
            rank = next((i+1 for i, s in enumerate(scores) if s["name"] == subj), len(scores))
            ranks.append(rank)
            print(f"    Subject '{subj}' ranked: {rank} / {len(scores)}")
        
        return ranks
        
    except Exception as e:
        print(f"  [Attack] Error during AlphaEdit attack: {e}")
        return []


def attack_rome(model, tok, case_id, layer, knowledge_template, module_template, true_subjects, ds_name="mcf"):
    """ROME 攻击评估
    
    Args:
        true_subjects: 真实被编辑的主语列表
        ds_name: 数据集名称
    """
    print(f"  [Attack] Running ROME attack for case {case_id}...")
    print(f"  [Attack] True subjects: {true_subjects}")
    
    edit_amounts_path = f"rome_orth_edit_amounts/edit_amounts_case_{case_id}.pt"
    
    if not Path(edit_amounts_path).exists():
        print(f"  [Attack] Warning: File not found: {edit_amounts_path}")
        return []
    
    try:
        edit_amounts = torch.load(edit_amounts_path)
        weight_key = f"{module_template.format(layer)}.weight"
        
        matching = [k for k in edit_amounts.keys() if str(layer) in k]
        if matching:
            weight_key = matching[0]
        else:
            print(f"  [Attack] Warning: Layer data not found")
            return []
        
        layer_data = edit_amounts[weight_key]
        upd_matrix = layer_data["upd_matrix"].cuda().double()
        
        # Get C matrix
        from rome.compute_u import get_cov as get_rome_cov
        from types import SimpleNamespace
        
        config = SimpleNamespace(
            model_name=model.config._name_or_path,
            mom2_dataset="wikipedia",
            mom2_n_samples=100000,
            mom2_dtype="float32",
            rewrite_module_tmp=module_template,
        )
        
        C = get_rome_cov(
            model, tok, module_template.format(layer),
            config.mom2_dataset, config.mom2_n_samples,
            config.mom2_dtype, hparams=config
        ).double()
        
        # SVD for k_post
        U, S, Vh = torch.linalg.svd(upd_matrix, full_matrices=False)
        k_post = Vh[0]
        k_pre_rec = C @ k_post
        
        # Score candidates
        import torch.nn.functional as F
        name_db = get_candidate_subjects(ds_name)
        scores = []
        k_pre_rec_norm = F.normalize(k_pre_rec, dim=0)
        
        for name in name_db:
            try:
                k_cand = get_activation_vector(
                    model, tok, name, knowledge_template, layer, module_template
                ).cuda().double()
                
                k_cand_norm = F.normalize(k_cand, dim=0)
                sim = torch.dot(k_pre_rec_norm, k_cand_norm)
                scores.append({"name": name, "score": abs(sim.item())})
            except:
                continue
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Find ranks of true subjects
        ranks = []
        for subj in true_subjects:
            rank = next((i+1 for i, s in enumerate(scores) if s["name"] == subj), len(scores))
            ranks.append(rank)
            print(f"    Subject '{subj}' ranked: {rank} / {len(scores)}")
        
        return ranks
        
    except Exception as e:
        print(f"  [Attack] Error during ROME attack: {e}")
        return []


def run_experiment(
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    dataset_size_limit: int,
    num_edits: int,
    n_independent_runs: int,
    camouflage_scales: List[int],
    ds_name: str = "mcf",
):
    """运行 camouflage_scale 消融实验"""
    
    print("="*80)
    print(f"Camouflage Scale Ablation Experiment: {alg_name}, Dataset: {ds_name}")
    print("="*80)
    
    # Get algorithm and params
    params_class, apply_algo = ALG_DICT[alg_name]
    
    # Determine hparams directory
    if "MEMIT" in alg_name:
        params_path = HPARAMS_DIR / "MEMIT" / hparams_fname
    elif "ROME" in alg_name:
        params_path = HPARAMS_DIR / "ROME" / hparams_fname
    elif "AlphaEdit" in alg_name:
        params_path = HPARAMS_DIR / "AlphaEdit" / hparams_fname
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    
    # Load base hyperparameters
    base_hparams = params_class.from_json(params_path)
    print(f"Loaded base parameters from {params_path}")
    
    # Load dataset
    print(f"Loading dataset {ds_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    
    if ds_name == "mcf":
        ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=dataset_size_limit)
    elif ds_name == "zsre":
        ds = MENDQADataset(DATA_DIR, tok=tok, size=dataset_size_limit)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    ds_list = list(ds)
    
    # Load evaluation dependencies (global for all runs)
    print(f"Loading evaluation dependencies for {ds_name}...")
    snips = None
    vec = None
    if ds_name == "mcf":
        snips = AttributeSnippets(DATA_DIR)
        vec = get_tfidf_vectorizer(DATA_DIR)
    
    # Determine num_edits per run
    if "ROME" in alg_name:
        actual_num_edits = 1  # ROME only edits one at a time
    else:
        actual_num_edits = num_edits
    
    # Initialize P matrix cache (only for AlphaEdit)
    P = None

    # Main experiment loop
    for scale in camouflage_scales:
        print(f"\n{'='*80}")
        print(f"Camouflage Scale = {scale}")
        print(f"{'='*80}")
        
        # Create result directory in main_ablation_experiments/results/
        # scale_dir = Path("main_ablation_experiments/results") /"zsre"/ f"camouflage_scale={scale}" /alg_name
        scale_dir = Path("main_ablation_experiments/results") / f"camouflage_scale={scale}" / alg_name
        scale_dir.mkdir(parents=True, exist_ok=True)
        
        # Store all ranks for this scale
        all_ranks_this_scale = []
        
        # Run n independent edits
        for run_idx in range(n_independent_runs):
            print(f"\n[Run {run_idx+1}/{n_independent_runs}]")
            
            # Create fresh model for this run
            print("  Loading fresh model...")
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
            
            # Sample data
            import random
            sampled_records = random.sample(ds_list, actual_num_edits)
            
            # Modify hparams with current camouflage_scale
            hparams = params_class.from_json(params_path)
            # 直接设置 camouflage_scale 属性（Python 的动态特性允许这样做）
            hparams.camouflage_scale = scale
            print(f"  Set camouflage_scale = {scale} in hparams")
            
            # Compute P if needed (lazy initialization using the first loaded model)
            if "AlphaEdit" in alg_name and P is None:
                print("\n  Computing P matrices for AlphaEdit (Lazy Init)...")
                W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
                
                # Determine dimension
                # GPT-2 weights are transposed (out, in) -> (in, out) in some contexts, but nethook usually returns (out, in) for Linear
                # Wait, standard pytorch Linear is (out, in).
                # The original code used shape[0] for gpt2-xl and shape[1] for others.
                # Let's stick to the original logic to be safe.
                if hparams.model_name == "gpt2-xl":
                    dim = W_out.shape[0]
                elif hparams.model_name in ["EleutherAI_gpt-j-6B", "Llama3-8B", "phi-1.5"]:
                    dim = W_out.shape[1]
                else:
                    dim = W_out.shape[1]
                
                P = torch.zeros((len(hparams.layers), dim, dim), device="cpu")
                del W_out
                
                for i, layer in enumerate(hparams.layers):
                    print(f"    Computing P for layer {layer}...")
                    P[i,:,:] = get_project(model, tok, layer, hparams)
                
                print("  P matrices computed and cached.")
                torch.cuda.empty_cache()

            # Apply editing
            print(f"  Applying {alg_name} with camouflage_scale={scale}...")
            try:
                if "AlphaEdit" in alg_name:
                    # Use pre-computed P matrix
                    # Reset cache_c for this run
                    W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
                    if hparams.model_name == "gpt2-xl":
                        cache_c_run = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
                    elif hparams.model_name in ["EleutherAI_gpt-j-6B", "Llama3-8B", "phi-1.5"]:
                        cache_c_run = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                    else:
                        cache_c_run = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                    del W_out
                    
                    edited_model, cache_c_run = apply_algo(
                        model, tok,
                        [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records],
                        hparams,
                        cache_c=cache_c_run,
                        P=P,
                    )
                else:
                    edited_model, _ = apply_algo(
                        model, tok,
                        [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records],
                        hparams,
                    )
                
                print(f"  Editing completed.")
                
            except Exception as e:
                print(f"  Error during editing: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Dataset Evaluation
            run_dir = scale_dir / f"edit{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Running {ds_name} evaluation...")
            
            try:
                # Get evaluation method for the dataset
                eval_method = DS_EVAL_METHOD_MAP.get(ds_name)
                if eval_method is None:
                    raise ValueError(f"No evaluation method found for dataset: {ds_name}")
                
                # Evaluate each case
                case_ids = [r["case_id"] for r in sampled_records]
                case_result_template = str(run_dir / "{}_edits-case_{}.json")
                
                for record in sampled_records:
                    case_id = record["case_id"]
                    out_file = Path(case_result_template.format(actual_num_edits, case_id))
                    
                    # Skip if already exists
                    if out_file.exists():
                        print(f"    Loading existing evaluation result for case {case_id}")
                    else:
                        # Evaluate
                        metrics = {
                            "case_id": case_id,
                            "grouped_case_ids": case_ids,
                            "num_edits": actual_num_edits,
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
                        print(f"    Evaluated and saved case {case_id}")
            except Exception as e:
                print(f"  Error during {ds_name} evaluation: {e}")
                import traceback
                traceback.print_exc()

            # GLUE Evaluation
            print(f"  Running GLUE evaluation...")
            glue_results = {'edit_num': run_idx, 'camouflage_scale': scale}
            out_file = str(run_dir / "glue_results.json")
            
            # 手动清理 CUDA 缓存
            torch.cuda.empty_cache()
            
            try:
                glue_eval = GLUEEval(model, tok, number_of_tests=100)
                glue_results = glue_eval.evaluate(
                    glue_results, out_file,
                    nli_flag=True, sst_flag=True, cola_flag=True,
                    rte_flag=True, mmlu_flag=True, mrpc_flag=True
                )
                
                with open(out_file, "w") as f:
                    json.dump(glue_results, f, indent=4)
                    
                print(f"  GLUE results saved to {out_file}")
                
            except Exception as e:
                print(f"  Error during GLUE evaluation: {e}")
            
            # Privacy Attack Evaluation
            print(f"  Running privacy attack...")
            case_ids = [r["case_id"] for r in sampled_records]
            # Extract true subjects from sampled records
            true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]
            
            knowledge_template = "The mother tongue of {} is"  # May need to adjust per dataset
            module_template = hparams.rewrite_module_tmp
            target_layer = hparams.layers[0] if hparams.layers else 4

            # 使用第一个 case_id 作为批次文件的 case_id
            batch_file_case_id = case_ids[0]

            # 手动清理 CUDA 缓存
            torch.cuda.empty_cache()

            # 一次性传入所有真实主语，只调用一次攻击函数
            try:
                if "MEMIT" in alg_name:
                    target_rank = actual_num_edits
                    ranks_this_run = attack_memit(
                        model, tok, batch_file_case_id, target_rank,
                        target_layer, knowledge_template, module_template,
                        true_subjects,  # 传入所有主语
                        ds_name=ds_name
                    )
                elif "AlphaEdit" in alg_name:
                    target_rank = actual_num_edits
                    ranks_this_run = attack_alphaedit(
                        model, tok, batch_file_case_id, target_rank,
                        target_layer, knowledge_template, module_template,
                        true_subjects,  # 传入所有主语
                        ds_name=ds_name
                    )
                elif "ROME" in alg_name:
                    ranks_this_run = attack_rome(
                        model, tok, batch_file_case_id, target_layer,
                        knowledge_template, module_template,
                        true_subjects,  # 传入所有主语
                        ds_name=ds_name
                    )
                else:
                    ranks_this_run = []
                
                # Calculate average rank for this run
                if ranks_this_run:
                    avg_rank_this_run = np.mean(ranks_this_run)
                    print(f"  Average rank for this run: {avg_rank_this_run:.2f}")
                    all_ranks_this_scale.append(avg_rank_this_run)
                    
            except Exception as e:
                print(f"    Error during attack: {e}")
                import traceback
                traceback.print_exc()

            # 1. 删除变量引用
            if 'model' in locals(): del model
            if 'edited_model' in locals(): del edited_model
            
            # 删除 AlphaEdit 特有的缓存
            if 'cache_c_run' in locals(): del cache_c_run

            # 2. 强制回收内存 (Python RAM)
            import gc
            gc.collect()

            # Clear CUDA cache
            torch.cuda.empty_cache()
        
        # Calculate and save overall privacy rank for this scale
        if all_ranks_this_scale:
            overall_avg_rank = np.mean(all_ranks_this_scale)
            privacy_results = {
                'camouflage_scale': scale,
                'individual_run_avg_ranks': all_ranks_this_scale,
                'overall_average_rank': overall_avg_rank,
                'n_runs': len(all_ranks_this_scale)
            }
            
            privacy_file = scale_dir / "privacy_rank.json"
            with open(privacy_file, "w") as f:
                json.dump(privacy_results, f, indent=4)
            
            print(f"\n[Scale {scale}] Overall Average Rank: {overall_avg_rank:.2f}")
            print(f"Privacy results saved to {privacy_file}")


def main():
    parser = argparse.ArgumentParser(description="Camouflage Scale Ablation Experiment")
    
    parser.add_argument("--alg_name", type=str, required=True,
                       help="Defence algorithm name (comma-separated list supported, e.g., ROME_defence,MEMIT_defence)")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Model name")
    parser.add_argument("--hparams_fname", type=str,
                       default="Llama3-8B.json",
                       help="Hyperparameters filename")
    parser.add_argument("--dataset_size_limit", type=int,
                       default=2000,
                       help="Dataset size limit")
    parser.add_argument("--num_edits", type=int,
                       default=10,
                       help="Number of edits per run (ignored for ROME)")
    parser.add_argument("--n_independent_runs", type=int,
                       default=5,
                       help="Number of independent editing runs per scale")
    parser.add_argument("--scales", type=str,
                       default="1,2,3,4,5,6,7,8,9,10",
                       help="Comma-separated list of camouflage scales to test")
    parser.add_argument("--ds_name", type=str,
                       default="mcf",
                       choices=["mcf", "zsre"],
                       help="Dataset name (mcf or zsre)")
    
    args = parser.parse_args()
    
    # Parse scales
    camouflage_scales = [float(s) for s in args.scales.split(",")]
    
    # Parse alg_names
    alg_names = args.alg_name.split(",")
    valid_algs = ["ROME_defence", "MEMIT_defence", "AlphaEdit_defence"]
    
    for alg in alg_names:
        if alg not in valid_algs:
            raise ValueError(f"Invalid algorithm name: {alg}. Choices are {valid_algs}")
            
        print(f"\nRunning experiment for algorithm: {alg}")
        run_experiment(
            alg_name=alg,
            model_name=args.model_name,
            hparams_fname=args.hparams_fname,
            dataset_size_limit=args.dataset_size_limit,
            num_edits=args.num_edits,
            n_independent_runs=args.n_independent_runs,
            camouflage_scales=camouflage_scales,
            ds_name=args.ds_name,
        )


if __name__ == "__main__":
    main()
