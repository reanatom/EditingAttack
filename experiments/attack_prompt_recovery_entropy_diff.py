import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from pathlib import Path
import os
import random
from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT
from experiments.attack_memit_recovery import get_activation_vector_for_name, compute_similarity_metrics
from util import nethook
from util.data_loader import load_dataset_data


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

    print(f"Loading edit amounts from {delta_file_path}...")
    try:
        edit_amounts = torch.load(delta_file_path)
    except Exception as e:
        print(f"Error loading edits: {e}")
        return False

    with torch.no_grad():
        for w_name, upd_matrix in edit_amounts.items():

            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix.to(w.device)
            

            
            w[...] += upd_matrix.float()
            
    print(f"Applied edits to {len(edit_amounts)} layers.")
    return True

def calculate_prediction_entropy(model, tok, prompt, subject):

    full_prompt = prompt.format(subject)
    input_ids = tok(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**input_ids)

    logits = outputs.logits[0, -1, :]


    probs = torch.softmax(logits, dim=-1)


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
    

    TRUE_PROMPT_TEMPLATE = "{}, the"
    # ====================================
    

    case_id = "6"
    config_rec = SimpleNamespace(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        delta_file_path=f"multi_case_edit_memit_amount_attack/edit_amounts_batch_case_{case_id}.pt",
        kr_ground_truth_path=f"multi_case_edit_memit_amount_attack/kr_ground_truth_case_{case_id}.pt",
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        final_layer_to_attack=4,
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
        

    print("\n[Step 1] Recovering top subject...")
    try:

        opt_k_4 = run_attack_simple_k_r_CORRECT(config_rec, model, tok)

    except Exception as e:
        print(f"Error getting optimized k: {e}")
        return


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
    

    prompt_lib = create_prompt_library(best_subject)
    print(f"Generated {len(prompt_lib)} prompts.")
    

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
            

    print("\n[Step 3] Applying edits to model...")
    if not apply_edits_to_model(model, config_rec.delta_file_path):
        print("Failed to apply edits.")
        return


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


    prompt_scores.sort(key=lambda x: x['score'], reverse=True)


    print("\n" + "=" * 100)
    print("[Result] Top 25 Prompts by Entropy-Diff Score")
    print("=" * 100)
    print(f"{'Rank':<6} {'Score':<12} {'E_unedit':<10} {'E_edit':<10} {'Prompt'}")
    print("-" * 120)

    top_k = 25
    for rank, item in enumerate(prompt_scores[:top_k], 1):
        print(
            f"{rank:<6} {item['score']:<12.6f} {item['e_unedit']:<10.6f} {item['e_edit']:<10.6f} {item['prompt']}"
        )
    print("-" * 120)

if __name__ == "__main__":
    main()

