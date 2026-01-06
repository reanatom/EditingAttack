import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from pathlib import Path
import os
import random
from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT
from experiments.attack_memit_recovery import get_activation_vector_for_name, compute_similarity_metrics
from util import nethook
from util.data_loader import load_dataset_data

# 尝试导入spacy
try:
    import spacy
    import subprocess
    import sys
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spacy not available. Token filtering will be disabled.")

# 尝试导入sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic similarity calculation will be disabled.")

# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ds_name ="mcf"
def create_prompt_library(subject):
    """
    Build a prompt library using prompts from the first 2000 cases of multi_counterfact.json.
    """
    _, prompts = load_dataset_data(ds_name=ds_name, limit=2000)
    
    # Ensure true prompt is in the list
    true_prompt = "The mother tongue of {} is"
    if true_prompt not in prompts:
        prompts.append(true_prompt)
        
    return prompts

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


def is_meaningful_token(token, nlp_model):
    """
    使用Spacy判断token是否是有意义的词类（专有名词、名词、数字）
    
    Args:
        token: 要判断的token字符串
        nlp_model: spacy模型
    
    Returns:
        bool: 如果是有意义的词类返回True，否则返回False
    """
    if not SPACY_AVAILABLE or nlp_model is None:
        return True  # 如果spacy不可用，不过滤
    
    # 处理空字符串
    if not token or not token.strip():
        return False
    
    try:
        # 使用spacy处理token
        doc = nlp_model(token)
        
        # 如果没有token，返回False
        if len(doc) == 0:
            return False
        
        # 获取第一个token的词性
        pos = doc[0].pos_
        
        # 保留的词类：PROPN（专有名词）、NOUN（名词）、NUM（数字）
        meaningful_pos = {'PROPN', 'NOUN', 'NUM'}
        
        # 检查是否是数字（纯数字字符串）
        if token.strip().replace('.', '').replace('-', '').isdigit():
            return True
        
        # 检查词性
        if pos in meaningful_pos:
            return True
        
        return False
    except Exception as e:
        # 如果处理出错，默认保留（避免过滤掉有效token）
        print(f"Warning: Error processing token '{token}' with spacy: {e}")
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

    return entropy


def create_name_database():
    """
    Create a name database using the first 2000 subjects from multi_counterfact.json
    """
    subjects, _ = load_dataset_data(ds_name=ds_name, limit=2000)

    # Ensure target name is in the list if not already (though if it's in the first 2000 it should be)
    target_name = "Danielle Darrieux"
    if target_name not in subjects:
        subjects.insert(0, target_name)

    return subjects[
           :2000]  # Return all loaded subjects (or limit if needed, user said "use ... to build database", implied use all 2000)

def main():
    print("=" * 80)
    print("Attack: Prompt Recovery using Entropy Difference (E_unedit - E_edit)")
    print("=" * 80)
    
    # ===== 用户配置：真实提示词模板 =====
    # 请在这里填写真实的提示词模板（包含{}占位符）
    TRUE_PROMPT_TEMPLATE = "{}, the"  # 请根据实际情况修改
    # ====================================
    
    # 1. 配置
    case_id = "2"
    config_rec = SimpleNamespace(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        delta_file_path=f"multi_case_edit_memit_amount_attack/edit_amounts_batch_case_{case_id}.pt",
        kr_ground_truth_path=f"multi_case_edit_memit_amount_attack/kr_ground_truth_case_{case_id}.pt",
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
        
    # 2. 获取 v1_r_direction_layer4 和 恢复主语 (这里主要为了恢复主语)
    print("\n[Step 1] Recovering top subject...")
    try:
        # 运行攻击以恢复 k 向量
        opt_k_4 = run_attack_simple_k_r_CORRECT(config_rec, model, tok)
        # 注意：此方法不使用 v1_r_direction
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
            model, tok, name, "The US dollars {} earn per year is", 
            config_rec.final_layer_to_attack, config_rec.rewrite_module_tmp, "subject_last"
        )
        sim, _ = compute_similarity_metrics(opt_k_4.squeeze(), vec)
        sim = abs(sim) 
        if sim > best_sim:
            best_sim = sim
            best_subject = name
            
    print(f"Recovered Top 1 Subject: {best_subject} (Sim: {best_sim:.4f})")
    
    # 3. 收集提示词
    prompt_lib = create_prompt_library(best_subject)
    print(f"Generated {len(prompt_lib)} prompts.")
    
    # 4. 在未编辑模型上计算熵 E_unedit
    print("\n[Step 2] Calculating Entropy on Unedited Model (E_unedit)...")
    
    unedit_entropies = {}
    
    for i, prompt in enumerate(prompt_lib):
        try:
            e_unedit = calculate_prediction_entropy(model, tok, prompt, best_subject)
            unedit_entropies[prompt] = e_unedit
            # if (i+1) % 20 == 0:
                # print(f"Processed {i+1} prompts on unedited model...")
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            
    # 5. 将模型转换为编辑后模型
    print("\n[Step 3] Applying edits to model...")
    if not apply_edits_to_model(model, config_rec.delta_file_path):
        print("Failed to apply edits.")
        return

    # # [新增功能] 打印真实Prompt在编辑后模型的预测 Top 10
    # print("\n[Debug] Checking predictions for True Prompt on Edited Model...")
    # # debug_prompt = "The official religion of Edwin of Northumbria is"
    # debug_prompt = "Which family does Ramalinaceae belong to? Answer:"
    # inputs = tok(debug_prompt, return_tensors="pt").to("cuda")
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     logits = outputs.logits[0, -1, :]
    #     probs = torch.softmax(logits, dim=-1)
    #     top_probs, top_indices = torch.topk(probs, 10)
    #
    # print(f"Prompt: '{debug_prompt}'")
    # print("Top 10 predicted tokens:")
    # for i in range(10):
    #     pred_token = tok.decode(top_indices[i])
    #     print(f"  {i+1}. '{pred_token}' (Prob: {top_probs[i].item():.4f})")

    # 6. 在编辑后模型上计算熵 E_edit 并计算得分
    print("\n[Step 4] Calculating Entropy on Edited Model (E_edit) and Score...")
    print("-" * 100)
    print(f"{'Rank':<4} {'Score (E_un / E_ed - 1)':<20} {'E_unedit':<10} {'E_edit':<10} {'Prompt'}")
    print("-" * 100)

    prompt_scores = []

    for i, prompt in enumerate(prompt_lib):
        if prompt not in unedit_entropies:
            continue
            
        try:
            e_edit = calculate_prediction_entropy(model, tok, prompt, best_subject)
            e_unedit = unedit_entropies[prompt]
            
            # 计算分数: E_unedit - E_edit
            score = (e_unedit - e_edit)/e_edit
            
            prompt_scores.append({
                "prompt": prompt,
                "e_unedit": e_unedit,
                "e_edit": e_edit,
                "score": score
            })
            
            # if (i+1) % 20 == 0:
                # print(f"Processed {i+1} prompts on edited model...")
                
        except Exception as e:
            print(f"Error evaluating prompt '{prompt}': {e}")
            continue

    # 7. 排序并打印前10名
    # 根据 Score 降序排序
    prompt_scores.sort(key=lambda x: x['score'], reverse=True)

    # # [检查特定提示词] 查找 "Which family does {} belong to?" 的排名和分数
    # target_prompt = "Which family does {} belong to?"
    # print("\n" + "=" * 100)
    # print(f"[Check] Searching for prompt: '{target_prompt}'")
    # print("=" * 100)
    #
    # target_rank = None
    # target_item = None
    # for rank, item in enumerate(prompt_scores, 1):
    #     if item['prompt'] == target_prompt:
    #         target_rank = rank
    #         target_item = item
    #         break
    #
    # if target_item:
    #     print(f"✓ Found! Rank: {target_rank}")
    #     print(f"  Prompt: '{target_prompt}'")
    #     print(f"  Score (E_un / E_ed - 1): {target_item['score']:.6f}")
    #     print(f"  E_unedit (unedited entropy): {target_item['e_unedit']:.6f}")
    #     print(f"  E_edit (edited entropy): {target_item['e_edit']:.6f}")
    # else:
    #     print(f"✗ Not found in prompt_scores list")
    #     # 检查是否在unedit_entropies中（可能因为某些原因没有被处理）
    #     if target_prompt in unedit_entropies:
    #         print(f"  Note: This prompt exists in unedit_entropies but was not processed in edited model")
    #         print(f"  E_unedit: {unedit_entropies[target_prompt]:.6f}")
    #     else:
    #         print(f"  Note: This prompt was not found in the prompt library")
    #
    # print("=" * 100)

    # # [新增功能] 获取前20个提示词，填充主语，输入编辑后模型，打印下一个token
    # print("\n" + "=" * 100)
    # print("[New Feature] Top 20 Prompts with Predicted Next Tokens")
    # print("=" * 100)
    # print(f"{'Rank':<6} {'Score':<15} {'Prompt':<50} {'Next Token':<30}")
    # print("-" * 100)
    #
    # top20_prompts = prompt_scores[:20]
    # for rank, item in enumerate(top20_prompts, 1):
    #     prompt_template = item['prompt']
    #     full_prompt = prompt_template.format(best_subject)
    #
    #     try:
    #         # 输入到编辑后的模型
    #         inputs = tok(full_prompt, return_tensors="pt").to("cuda")
    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #             logits = outputs.logits[0, -1, :]  # 获取最后一个token的logits
    #
    #         # 获取下一个token（top1）
    #         next_token_id = torch.argmax(logits, dim=-1).item()
    #         next_token = tok.decode([next_token_id]).strip()
    #
    #         print(f"{rank:<6} {item['score']:<15.4f} {prompt_template:<50} {next_token:<30}")
    #     except Exception as e:
    #         print(f"{rank:<6} {item['score']:<15.4f} {prompt_template:<50} Error: {e}")
    #
    # print("=" * 100)

    # [新增功能] 获取前10个提示词，在编辑前模型上测试并统计下一个token
    print("\n" + "=" * 100)
    print("[New Feature] Top 10 Prompts - Next Token Statistics on Unedited Model")
    print("=" * 100)
    
    # 加载spacy模型用于过滤
    nlp_model = None
    if SPACY_AVAILABLE:
        try:
            print("Loading spacy model for token filtering...")
            nlp_model = spacy.load("en_core_web_sm")
            print("Spacy model loaded successfully.")
        except OSError:
            print("en_core_web_sm model not found. Attempting to download...")
            try:
                # 自动下载模型
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                print("Model downloaded successfully. Loading...")
                nlp_model = spacy.load("en_core_web_sm")
                print("Spacy model loaded successfully.")
            except subprocess.CalledProcessError:
                print("Error: Failed to download en_core_web_sm model.")
                print("Please install it manually with: python -m spacy download en_core_web_sm")
                print("Token filtering will be disabled.")
            except Exception as e:
                print(f"Error loading model after download: {e}")
                print("Token filtering will be disabled.")
        except Exception as e:
            print(f"Warning: Error loading spacy model: {e}")
            print("Token filtering will be disabled.")
    
    # 保存前10个提示词
    top10_items = prompt_scores[:10]
    
    # 基于score过滤：丢掉score < 5的提示词
    score_threshold = 5.0
    filtered_top10_items = [item for item in top10_items if item['score'] >= score_threshold]
    top10_prompts = [item['prompt'] for item in filtered_top10_items]
    
    print(f"\n[Filtering] Top 10 prompts before score filtering: {len(top10_items)}")
    print(f"[Filtering] After filtering (score >= {score_threshold}): {len(filtered_top10_items)} prompts")
    if len(top10_items) > len(filtered_top10_items):
        filtered_out = [item for item in top10_items if item['score'] < score_threshold]
        print(f"[Filtering] Filtered out {len(filtered_out)} prompts with score < {score_threshold}:")
        for item in filtered_out:
            print(f"  - '{item['prompt']}' (score: {item['score']:.4f})")
    
    # 删除编辑后的模型以释放显存
    del model
    torch.cuda.empty_cache()
    
    # 重新加载未编辑的模型
    print("Reloading unedited model for testing top 10 prompts...")
    model_unedit = AutoModelForCausalLM.from_pretrained(config_rec.model_name).cuda()
    
    # 统计每个提示词预测的下一个token（只生成1个token）
    token_counts_all = {}  # {token: count} - 所有token
    token_counts_filtered = {}  # {token: count} - 过滤后的有意义的token
    prompt_token_map = []  # [(prompt, token, is_meaningful), ...] - 每个提示词对应一个token
    filtered_count = 0  # 被过滤掉的token数量
    
    for prompt_template in top10_prompts:
        full_prompt = prompt_template.format(best_subject)
        
        try:
            # 输入到未编辑的模型
            inputs = tok(full_prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model_unedit(**inputs)
                logits = outputs.logits[0, -1, :]  # 获取最后一个token的logits
                
                # 获取下一个token（top1）
                next_token_id = torch.argmax(logits, dim=-1).item()
                next_token = tok.decode([next_token_id]).strip()
            
            # 统计所有token
            if next_token not in token_counts_all:
                token_counts_all[next_token] = 0
            token_counts_all[next_token] += 1
            
            # 判断是否是有意义的token
            is_meaningful = is_meaningful_token(next_token, nlp_model)
            
            # 只统计有意义的token
            if is_meaningful:
                if next_token not in token_counts_filtered:
                    token_counts_filtered[next_token] = 0
                token_counts_filtered[next_token] += 1
            else:
                filtered_count += 1
            
            prompt_token_map.append((prompt_template, next_token, is_meaningful))
            
        except Exception as e:
            print(f"Error processing prompt '{prompt_template}': {e}")
            continue
    
    # 基于过滤后的token进行统计
    meaningful_prompts_count = len([x for x in prompt_token_map if x[2]])  # 有意义的token数量
    
    if token_counts_filtered:
        most_common_token = max(token_counts_filtered, key=token_counts_filtered.get)
        most_common_count = token_counts_filtered[most_common_token]
        percentage = (most_common_count / meaningful_prompts_count) * 100 if meaningful_prompts_count > 0 else 0
        
        print(f"\n[Filtered Statistics - Only Meaningful Tokens (PROPN, NOUN, NUM)]")
        print(f"Filtered out {filtered_count} meaningless tokens (articles, prepositions, verbs, pronouns, etc.)")
        print(f"Meaningful tokens: {meaningful_prompts_count} out of {len(top10_prompts)} prompts")
        print(f"\nMost common meaningful token: '{most_common_token}'")
        print(f"Occurrence count: {most_common_count} out of {meaningful_prompts_count} meaningful prompts")
        print(f"Percentage: {percentage:.1f}%")
        
        print("\nDetailed breakdown:")
        print(f"{'Prompt':<60} {'Next Token':<30} {'Meaningful':<12}")
        print("-" * 110)
        for prompt_template, next_token, is_meaningful in prompt_token_map:
            meaningful_mark = "✓" if is_meaningful else "✗"
            print(f"{prompt_template:<60} {next_token:<30} {meaningful_mark:<12}")
        
        print("\nMeaningful token frequency distribution:")
        for token, count in sorted(token_counts_filtered.items(), key=lambda x: x[1], reverse=True):
            pct = (count / meaningful_prompts_count) * 100 if meaningful_prompts_count > 0 else 0
            print(f"  '{token}': {count} times ({pct:.1f}%)")
        
        if filtered_count > 0:
            print("\nFiltered out tokens (not meaningful):")
            filtered_tokens = {token: count for token, count in token_counts_all.items() 
                             if token not in token_counts_filtered}
            for token, count in sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(top10_prompts)) * 100
                print(f"  '{token}': {count} times ({pct:.1f}%)")
        
        # [新增功能] 找出所有输出最常出现有意义token的提示词，使用SentenceBERT计算语义相似度
        print("\n" + "=" * 100)
        print("[New Feature] Semantic Similarity Analysis using SentenceBERT")
        print("=" * 100)
        
        # 找出所有输出最常出现有意义token的提示词
        candidate_prompts = []
        for prompt_template, next_token, is_meaningful in prompt_token_map:
            if is_meaningful and next_token == most_common_token:
                candidate_prompts.append(prompt_template)
        
        print(f"\nFound {len(candidate_prompts)} prompts that output the most common meaningful token '{most_common_token}':")
        for i, prompt in enumerate(candidate_prompts, 1):
            print(f"  {i}. {prompt}")
        
        if candidate_prompts and SENTENCE_TRANSFORMERS_AVAILABLE:
            # 使用用户配置的真实提示词模板
            true_prompt_template = TRUE_PROMPT_TEMPLATE
            
            print(f"\nTrue prompt template: '{true_prompt_template}'")
            print(f"Subject (recovered): '{best_subject}'")
            
            # 加载SentenceBERT模型
            print("\nLoading SentenceBERT model...")
            try:
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("SentenceBERT model loaded successfully.")
            except Exception as e:
                print(f"Error loading SentenceBERT model: {e}")
                sentence_model = None
            
            if sentence_model:
                # 填充恢复出来的best_subject到候选提示词和真实提示词
                candidate_prompts_filled = [prompt.format(best_subject) for prompt in candidate_prompts]
                true_prompt_filled = true_prompt_template.format(best_subject)
                
                print(f"\nCalculating semantic similarities...")
                print(f"{'Candidate Prompt':<70} {'Similarity':<15}")
                print("-" * 90)
                
                # 批量编码所有提示词
                all_prompts = candidate_prompts_filled + [true_prompt_filled]
                embeddings = sentence_model.encode(all_prompts, convert_to_tensor=True)
                
                # 分离候选提示词和真实提示词的embeddings
                candidate_embeddings = embeddings[:-1]  # 所有候选提示词
                true_embedding = embeddings[-1:]  # 真实提示词
                
                # 使用util.cos_sim计算相似度
                # 使用view(-1)确保即使只有一个候选提示词也能正确处理
                similarities = util.cos_sim(candidate_embeddings, true_embedding).squeeze().view(-1).tolist()
                
                # 打印每个候选提示词的相似度
                for idx, original_prompt in enumerate(candidate_prompts):
                    similarity = similarities[idx]
                    print(f"{original_prompt:<70} {similarity:.4f}")
                
                # 计算平均值
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                print("-" * 90)
                print(f"{'Average Similarity':<70} {avg_similarity:.4f}")
                print(f"\nNumber of candidate prompts: {len(candidate_prompts)}")
                print(f"Average semantic similarity with true prompt: {avg_similarity:.4f}")
            else:
                print("SentenceBERT model not available. Skipping semantic similarity calculation.")
        elif not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("\nWarning: sentence-transformers not available. Please install it with: pip install sentence-transformers")
        else:
            print(f"\nNo candidate prompts found that output the most common meaningful token '{most_common_token}'.")
    else:
        print("No meaningful tokens were found after filtering.")
        print("\nAll tokens (before filtering):")
        for token, count in sorted(token_counts_all.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(top10_prompts)) * 100
            print(f"  '{token}': {count} times ({pct:.1f}%)")
    
    print("=" * 100)
    
    # 清理未编辑模型（后续不需要模型了）
    del model_unedit
    torch.cuda.empty_cache()

    true_prompt = "The mother tongue of {} is"
    found_true = False

    for i, item in enumerate(prompt_scores[:10], 1):
        is_true = "✓" if item['prompt'] == true_prompt else " "
        print(
            f"{i:<4} {item['score']:<20.4f} {item['e_unedit']:<10.4f} {item['e_edit']:<10.4f} {item['prompt']} {is_true}"
        )
        if item['prompt'] == true_prompt:
            found_true = True

    print("-" * 100)

    if found_true:
        print("\n🎉 SUCCESS: True prompt found in Top 10 based on Entropy Diff!")
    else:
        print("\n⚠️ Note: True prompt not in Top 10.")

    # 打印真实提示词的排名
    print("\nChecking True Prompt rank based on Entropy Diff...")
    rank = next((i+1 for i,item in enumerate(prompt_scores) if item["prompt"]==true_prompt), None)
    
    if rank:
        item = next((item for item in prompt_scores if item["prompt"]==true_prompt), None)
        print(f"True Prompt: '{true_prompt}'")
        print(f"Score (E_un / E_ed - 1): {item['score']:.4f}")
        print(f"E_unedit: {item['e_unedit']:.4f}")
        print(f"E_edit: {item['e_edit']:.4f}")
        print(f"Rank: {rank}")
    else:
        print(f"True Prompt '{true_prompt}' not processed or error occurred.")

if __name__ == "__main__":
    main()

