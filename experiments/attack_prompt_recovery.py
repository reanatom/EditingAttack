import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from pathlib import Path
import os
import random
from memit.compute_z import get_module_input_output_at_words
from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT
from experiments.attack_memit_recovery import create_name_database, get_activation_vector_for_name, compute_similarity_metrics
from util import nethook
from memit.memit_main import upd_matrix_match_shape

# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.data_loader import load_dataset_data

def create_prompt_library(subject):
    """
    Build a prompt library using prompts from the first 2000 cases of multi_counterfact.json.
    """
    _, prompts = load_dataset_data(limit=2000)
    
    # Ensure true prompt is in the list
    true_prompt = "The mother tongue of {} is"
    if true_prompt not in prompts:
        prompts.append(true_prompt)
        
    return prompts

def get_layer_output_activation(model, tok, prompt, subject, layer, module_template):
    """
    获取在主语最后一个token位置上的投影层（FFN down_proj）输出的激活向量。
    """
    context_templates = [prompt]
    words = [subject]
    
    # track="both" 在 get_module_input_output_at_words 中返回 (input, output)
    _, activation_output = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy="subject_last",
    )
    
    if activation_output.dim() > 1:
        activation_output = activation_output.squeeze(0)
        
    return activation_output.detach().cpu() # Move to CPU to save GPU memory

def apply_edits_to_model(model, delta_file_path):
    """
    加载编辑量并应用到模型。
    """
    print(f"Loading edit amounts from {delta_file_path}...")
    try:
        edit_amounts = torch.load(delta_file_path)
    except Exception as e:
        print(f"Error loading edits: {e}")
        return False

    with torch.no_grad():
        for w_name, upd_matrix in edit_amounts.items():
            # upd_matrix 已经在 memit_main 中被处理过形状并转为 CPU 了
            # 我们需要将其移回 GPU 并应用
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix.to(w.device)
            
            # 确保形状匹配 (虽然 memit_main 应该已经处理过了，但再次检查是个好习惯)
            # upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape) 
            # edit_amounts 保存的是已经 match_shape 过的
            
            w[...] += upd_matrix.float()
            
    print(f"Applied edits to {len(edit_amounts)} layers.")
    return True


def calculate_prediction_entropy(model, tok, prompt, subject):
    """
    计算模型对下一个词预测的熵 (Entropy)。
    熵越低，表示模型越自信；熵越高，表示模型越困惑。
    """
    full_prompt = prompt.format(subject)
    input_ids = tok(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**input_ids)

    # 获取最后一个 token 的 logits
    logits = outputs.logits[0, -1, :]

    # 计算概率分布
    probs = torch.softmax(logits, dim=-1)

    # 计算熵: -sum(p * log(p))
    # 为了数值稳定性，使用 log_softmax
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum().item()

    # 同时获取最大概率的词，方便观察
    max_prob, max_id = torch.max(probs, dim=-1)
    predicted_token = tok.decode(max_id).strip()

    return entropy, max_prob.item(), predicted_token

def calculate_score(entropy, max_prob, predicted_token):
    """
    计算最终得分：Score = Information Value / Entropy
    Information Value = max_prob * indicator
    indicator: 如果是无意义词则为0，否则为1
    """
    # 停用词列表
    stopwords = [
        "", " ", "<|endoftext|>", "the", "a", "an", "is", "of", "in", "to", "and", "that", "it", "for", "on", "with", "as", "by", "at", 
        "this", "these", "those", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
        "do", "does", "did", "but", "if", "or", "because", "so", "not", "no", "yes", 
        "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", 
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how", 
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "nor", "too", "very", 
        "can", "will", "just", "should", "now", "unk", "unknown"
    ]
    
    predicted_token_lower = predicted_token.lower().strip()
    
    # 指示函数
    indicator = 0 if predicted_token_lower in stopwords or len(predicted_token_lower) == 0 else 1
    
    # 避免除以零
    epsilon = 1e-6
    score = (max_prob * indicator) / (entropy + epsilon)
    
    return score, indicator

def main():
    print("=" * 80)
    print("Attack: Prompt Recovery using Entropy and Information Value")
    print("=" * 80)
    
    # 1. 配置
    case_id = "1"
    config_rec = SimpleNamespace(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        delta_file_path=f"edit_memit_amount/edit_amounts_batch_case_{case_id}.pt",
        kr_ground_truth_path=f"edit_memit_amount/kr_ground_truth_case_{case_id}.pt",
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        final_layer_to_attack=4, # 使用第4层进行主语恢复和获取 R
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
        mom2_update_weight=15000,
        learning_rate_r=1e-2,
        iterations=0,
        beta=5e-6,
        lambda_l2=5e-6,
    )

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
        
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config_rec.model_name).cuda()
    tok = AutoTokenizer.from_pretrained(config_rec.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    # 2. 获取 v1_r_direction_layer4 和 恢复主语
    print("\n[Step 1] Recovering top subject using Layer 4...")
    try:
        opt_k_4 = run_attack_simple_k_r_CORRECT(config_rec, model, tok)
        print("Successfully extracted k vector from Layer 4")
        # Note: v1_r_direction is no longer returned by the function
        opt_k_4 = opt_k_4.detach().cpu().squeeze()
    except Exception as e:
        print(f"Error getting optimized k: {e}")
        return

    # 创建名字数据库并找到最佳主语
    name_db = create_name_database()
    best_subject = None
    best_sim = -1.0
    
    print("Finding best matching subject...")
    for name in name_db:
        vec = get_activation_vector_for_name(
            model, tok, name, "The height of {} is", 
            config_rec.final_layer_to_attack, config_rec.rewrite_module_tmp, "subject_last"
        )
        sim, _ = compute_similarity_metrics(opt_k_4.squeeze(), vec)
        sim = abs(sim) # Apply user fix
        if sim > best_sim:
            best_sim = sim
            best_subject = name
            
    print(f"Recovered Top 1 Subject: {best_subject} (Sim: {best_sim:.4f})")
    
    # 3. 收集提示词
    prompt_lib = create_prompt_library(best_subject)
    print(f"Generated {len(prompt_lib)} prompts.")
    
    # 4. 将模型转换为编辑后模型
    print("\n[Step 2] Applying edits to model (for entropy calculation)...")
    if not apply_edits_to_model(model, config_rec.delta_file_path):
        print("Failed to apply edits.")
        return

    # 5. 评估提示词 (基于熵和信息价值)
    print("\n[Step 3] Evaluating prompts using Score = InfoValue / Entropy...")
    print("-" * 120)
    print(f"{'Rank':<4} {'Score':<10} {'Entropy':<10} {'Max Prob':<10} {'IV':<5} {'Pred':<15} {'Prompt'}")
    print("-" * 120)

    prompt_scores_entropy = []

    for i, prompt in enumerate(prompt_lib):
        try:
            entropy, max_prob, pred = calculate_prediction_entropy(
                model, tok, prompt, best_subject
            )
            
            score, iv_indicator = calculate_score(entropy, max_prob, pred)
            
            prompt_scores_entropy.append({
                "prompt": prompt,
                "entropy": entropy,
                "max_prob": max_prob,
                "pred_token": pred,
                "score": score,
                "iv_indicator": iv_indicator
            })
            
            if (i+1) % 20 == 0:
                print(f"Processed {i+1} prompts for entropy evaluation...")
                
        except Exception as e:
            print(f"Error evaluating prompt '{prompt}': {e}")
            continue

    # 根据 Score 降序排序
    prompt_scores_entropy.sort(key=lambda x: x['score'], reverse=True)

    true_prompt = "The mother tongue of {} is"
    found_true = False

    for i, item in enumerate(prompt_scores_entropy[:10], 1):
        is_true = "✓" if item['prompt'] == true_prompt else " "
        print(
            f"{i:<4} {item['score']:<10.4f} {item['entropy']:<10.4f} {item['max_prob']:<10.4f} {item['iv_indicator']:<5} {item['pred_token']:<15} {item['prompt']} {is_true}"
        )
        if item['prompt'] == true_prompt:
            found_true = True

    print("-" * 120)

    if found_true:
        print("\n🎉 SUCCESS: True prompt found in Top 10 based on Score!")
    else:
        print("\n⚠️ Note: True prompt not in Top 10.")

    # 打印真实提示词的排名
    print("\nChecking True Prompt rank based on Score...")
    rank = next((i+1 for i,item in enumerate(prompt_scores_entropy) if item["prompt"]==true_prompt), None)
    
    if rank:
        item = next((item for item in prompt_scores_entropy if item["prompt"]==true_prompt), None)
        print(f"True Prompt: '{true_prompt}'")
        print(f"Score: {item['score']:.4f}")
        print(f"Entropy: {item['entropy']:.4f}")
        print(f"Rank: {rank}")
    else:
        print(f"True Prompt '{true_prompt}' not processed or error occurred.")

if __name__ == "__main__":
    main()
