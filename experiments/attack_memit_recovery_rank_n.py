import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from pathlib import Path
import os
import sys


sys.path.append(os.getcwd())

from memit.compute_z import get_module_input_output_at_words
from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.data_loader import load_dataset_data

def create_name_database():

    subjects, _ = load_dataset_data(limit=2000)
    

    target_names_to_ensure = ["Danielle Darrieux", "Edwin of Northumbria"] 
    for name in target_names_to_ensure:
        if name not in subjects:
            subjects.insert(0, name)
            
    return subjects

def get_activation_vector_for_name(
    model, tok, name, template, layer, module_template, fact_token_strategy
):

    context_templates = [template]
    words = [name]
    

    activation_input, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy=fact_token_strategy,
    )
    

    if activation_input.dim() > 1:
        activation_input = activation_input.squeeze()
    
    return activation_input


def get_activation_vector_for_prompt(
    model, tok, subject, prompt_template, layer, module_template, fact_token_strategy
):


    full_prompt = prompt_template.format(subject)
    

    context_templates = [prompt_template]
    words = [subject]
    
    activation_input, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy=fact_token_strategy,
    )
    

    if activation_input.dim() > 1:
        activation_input = activation_input.squeeze()
    
    return activation_input




def main():
    print("=" * 80)
    print("Rank-N MEMIT Attack: Subject Recovery via Subspace Projection")
    print("=" * 80)

    # 1. 配置参数
    case_id = "0"
    target_rank = 10
    

    delta_path = f"multi_case_edit_memit_amount_attack/edit_amounts_batch_case_{case_id}.pt"
    gt_path = f"multi_case_edit_memit_amount_attack/kr_ground_truth_case_{case_id}.pt"

    config = SimpleNamespace(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        delta_file_path=delta_path,
        kr_ground_truth_path=gt_path,
        
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        final_layer_to_attack=4,
        
        target_rank=target_rank,
        

        # mom2_dataset="wikipedia",
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
        mom2_update_weight=15000,
        

        learning_rate_r=1e-2,
        iterations=0,
        beta=5e-6,
        lambda_l2=5e-6,
    )


    # knowledge_template = "The mother tongue of {} is"
    knowledge_template = "The US dollar {} earns each year is"
    

    if not Path(config.delta_file_path).exists():
        print(f"Error: Cannot find edit amount file: {config.delta_file_path}")

        return

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name).cuda()
    tok = AutoTokenizer.from_pretrained(config.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token


    print(f"\n[Step 1] Recovering subspace basis (Rank={target_rank})...")
    try:

        Q_basis = run_attack_simple_k_r_CORRECT(config, model, tok)
        
        print(f"Recovered Basis Shape: {Q_basis.shape}")
        
    except Exception as e:
        print(f"Error recovering basis: {e}")
        import traceback
        traceback.print_exc()
        return


    print("\n[Step 2] Scoring candidate subjects via projection...")
    name_database = create_name_database()
    scores = []
    
    Q_basis = Q_basis.to("cuda").double()

    for i, name in enumerate(name_database):
        try:

            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, 
                config.final_layer_to_attack, 
                config.rewrite_module_tmp, 
                "subject_last"
            ).to("cuda").double()
            


            proj_coeffs = Q_basis.T @ k_cand # (Rank,)
            proj_norm = torch.norm(proj_coeffs)
            cand_norm = torch.norm(k_cand)
            
            score = proj_norm / (cand_norm + 1e-9)
            
            scores.append({
                "name": name,
                "score": score.item(),
                "proj_norm": proj_norm.item(),
                "total_norm": cand_norm.item()
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(name_database)}...")

        except Exception as e:
            print(f"Error processing {name}: {e}")


    print("\n[Step 3] Top 10 Subjects by Projection Score:")
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"{'Rank':<6} {'Name':<30} {'Score':<15} {'Proj/Total Norm':<25}")
    print("-" * 80)
    for i, item in enumerate(scores[:10], 1):
        print(f"{i:<6} {item['name']:<30} {item['score']:<15.6f} {item['proj_norm']:.2f} / {item['total_norm']:.2f}")
    
    print("-" * 80)
    

    targets = ["Edwin of Northumbria", "Danielle Darrieux"]
    for t in targets:
        rank = next((i for i, s in enumerate(scores) if s["name"] == t), None)
        if rank is not None:
            print(f"Target '{t}' Rank: {rank + 1}, Score: {scores[rank]['score']:.6f}")
        else:
            print(f"Target '{t}' not found in database or results.")
    

    print("\n" + "=" * 80)
    print(f"[Step 4] Calculating prompt projection scores for top {target_rank} subjects")
    print("=" * 80)
    

    top_subjects = [item["name"] for item in scores[:target_rank]]
    print(f"\nTop {target_rank} subjects (participating in editing):")
    for i, subject in enumerate(top_subjects, 1):
        subject_score = next(item["score"] for item in scores if item["name"] == subject)
        print(f"  {i}. {subject} (score: {subject_score:.6f})")
    

    print("\nLoading prompt library...")
    _, prompt_library = load_dataset_data(limit=2000)
    print(f"Loaded {len(prompt_library)} prompts from dataset")
    
    Q_basis = Q_basis.to("cuda").double()
    

    for subject_idx, subject in enumerate(top_subjects, 1):
        print(f"\n{'=' * 80}")
        print(f"[Subject {subject_idx}/{len(top_subjects)}] {subject}")
        print("=" * 80)
        
        prompt_scores = []
        
        for prompt_idx, prompt_template in enumerate(prompt_library):
            try:

                activation_input = get_activation_vector_for_prompt(
                    model=model,
                    tok=tok,
                    subject=subject,
                    prompt_template=prompt_template,
                    layer=config.final_layer_to_attack,
                    module_template=config.rewrite_module_tmp,
                    fact_token_strategy="subject_last"
                ).to("cuda").double()
                

                proj_coeffs = Q_basis.T @ activation_input  # (Rank,)
                

                proj_norm = torch.norm(proj_coeffs)
                activation_norm = torch.norm(activation_input)
                score = proj_norm / (activation_norm + 1e-9)
                
                prompt_scores.append({
                    "prompt": prompt_template,
                    "score": score.item(),
                    "proj_norm": proj_norm.item(),
                    "activation_norm": activation_norm.item(),
                    "proj_coeffs": proj_coeffs.cpu().tolist()
                })
                
                if (prompt_idx + 1) % 100 == 0:
                    print(f"  Processed {prompt_idx + 1}/{len(prompt_library)} prompts...")
                    
            except Exception as e:
                print(f"  Error processing prompt '{prompt_template}' for subject '{subject}': {e}")
                continue
        

        prompt_scores.sort(key=lambda x: x["score"], reverse=True)
        

        print(f"\n[Subject: {subject}] Top prompts by projection score:")
        print(f"{'Rank':<6} {'Score':<15} {'Proj_Norm':<15} {'Activation_Norm':<15} {'Prompt':<60}")
        print("-" * 120)
        for rank, item in enumerate(prompt_scores[:20], 1):
            print(f"{rank:<6} {item['score']:<15.6f} {item['proj_norm']:<15.6f} {item['activation_norm']:<15.6f} {item['prompt']:<60}")
        
        print(f"\n[Subject: {subject}] Total prompts processed: {len(prompt_scores)}")
        if prompt_scores:
            print(f"  Highest score: {prompt_scores[0]['score']:.6f}")
            print(f"  Lowest score: {prompt_scores[-1]['score']:.6f}")
            print(f"  Average score: {sum(item['score'] for item in prompt_scores) / len(prompt_scores):.6f}")

if __name__ == "__main__":
    main()

