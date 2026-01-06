import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from pathlib import Path
import os
from memit.compute_z import get_module_input_output_at_words
from experiments.attack_kr_opt import run_attack_simple_k_r_CORRECT, nethook

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from util.data_loader import load_dataset_data

def create_name_database():
    """
    Create a name database using the first 2000 subjects from multi_counterfact.json
    """
    subjects, _ = load_dataset_data(limit=2000)
    
    # Ensure target name is in the list if not already (though if it's in the first 2000 it should be)
    target_name = "Danielle Darrieux"
    if target_name not in subjects:
        subjects.insert(0, target_name)
        
    return subjects[:2000] # Return all loaded subjects (or limit if needed, user said "use ... to build database", implied use all 2000)


def get_activation_vector_for_name(
    model, tok, name, template, layer, module_template, fact_token_strategy
):
    """
    Get the activation vector of the subject's last token at the target layer for the given name in the template
    """
    # Use get_module_input_output_at_words to get activation vector
    # We need to get the activation vector of the subject's last token
    context_templates = [template]
    words = [name]
    
    # Get input activation vector (input to MLP layer, i.e., k vector)
    activation_input, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy=fact_token_strategy,
    )
    
    # activation_input shape may be (1, hidden_dim) or (hidden_dim,)
    # Ensure return shape is (hidden_dim,)
    if activation_input.dim() > 1:
        activation_input = activation_input.squeeze(0)
    
    return activation_input  # Return (hidden_dim,)


def compute_similarity_metrics(vec1, vec2):
    """
    Compute similarity metrics between two vectors
    Returns: cosine similarity, MSE
    """
    # Ensure vectors are on the same device
    vec1 = vec1.to(vec2.device)
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1).item()
    
    # MSE
    mse = F.mse_loss(vec1, vec2).item()
    
    return cos_sim, mse


def test_model_output(model, tok, template, name):
    """
    Test model output for the given name
    Returns the generated text from the model
    """
    prompt = template.format(name)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    
    generated_text = tok.decode(outputs[0], skip_special_tokens=True)
    # Extract generated part (remove original prompt)
    generated_part = generated_text[len(prompt):].strip()
    return generated_part


def main():
    """
    Main attack function
    """
    print("=" * 80)
    print("MEMIT Attack: Recover edited subject and target word")
    print("=" * 80)
    
    # 1. Configuration parameters
    case_id = "1"
    config = SimpleNamespace(
        # Model and file paths
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        delta_file_path=f"orth_defence_edit_memit_amount/edit_amounts_batch_case_{case_id}.pt",
        kr_ground_truth_path=f"orth_defence_edit_memit_amount/kr_ground_truth_case_{case_id}.pt",
        
        # Layer template
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        final_layer_to_attack=4,  # Target layer
        
        # C matrix parameters
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
        
        # Solver parameters
        mom2_update_weight=15000,
        
        # Optimizer parameters
        learning_rate_r=1e-2,
        iterations=0,
        beta=5e-6,
        lambda_l2=5e-6,
    )
    
    # Knowledge template
    # knowledge_template = "The mother tongue of {} is"
    # knowledge_template = "The language {} primarily speaks of is"
    # knowledge_template = "The height of {} is"
    knowledge_template = "The US dollar {} earns each year is"
    target_subject = "Danielle Darrieux"
    target_answer = "French"  # Privacy knowledge being concealed
    
    # 2. Check if files exist
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    if not Path(config.delta_file_path).exists():
        print(f"Error: Cannot find edit amount file: {config.delta_file_path}")
        return
    
    print("CUDA available, starting to load model...")
    
    # 3. Load model and tokenizer
    print("\n[Step 1] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name).cuda()
    tok = AutoTokenizer.from_pretrained(config.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("Model and tokenizer loaded successfully")
    
    # 4. Call attack_kr_opt to get optimized k vector (subject's last token activation vector)
    print("\n[Step 2] Calling attack_kr_opt to get optimized k vector...")
    try:
        optimized_k = run_attack_simple_k_r_CORRECT(config, model, tok)
        print(f"Optimized k vector shape: {optimized_k.shape}")
        print(f"Optimized k vector norm: {torch.norm(optimized_k).item():.6f}")
    except Exception as e:
        print(f"Error when calling attack_kr_opt: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Create name database
    print("\n[Step 3] Creating name database...")
    name_database = create_name_database()
    print(f"Name database size: {len(name_database)}")
    print(f"Target name '{target_subject}' in database: {target_subject in name_database}")
    
    # 6. Enumerate name database, get activation vector for each name and compute similarity
    print("\n[Step 4] Enumerating name database, computing similarity...")
    similarity_scores = []
    
    # Get fact_token_strategy
    fact_token_strategy = "subject_last"  # Get subject's last token
    
    for idx, name in enumerate(name_database):
        try:
            # Get activation vector for this name
            activation_vec = get_activation_vector_for_name(
                model=model,
                tok=tok,
                name=name,
                template=knowledge_template,
                layer=config.final_layer_to_attack,
                module_template=config.rewrite_module_tmp,
                fact_token_strategy=fact_token_strategy,
            )
            
            # Compute similarity
            # optimized_k shape is (D_int, 1), need to squeeze
            optimized_k_vec = optimized_k.squeeze() if optimized_k.dim() > 1 else optimized_k
            cos_sim, mse = compute_similarity_metrics(optimized_k_vec, activation_vec)
            
            similarity_scores.append({
                "name": name,
                "cos_sim": abs(cos_sim),
                "mse": mse,
            })
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(name_database)} names...")
                
        except Exception as e:
            print(f"Error processing name '{name}': {e}")
            continue
    
    # 7. Sort by similarity and print top 10
    print("\n[Step 5] Top 10 names by similarity:")
    print("-" * 80)
    similarity_scores.sort(key=lambda x: x["cos_sim"], reverse=True)
    
    print(f"{'Rank':<6} {'Name':<30} {'Cosine Sim':<15} {'MSE':<15}")
    print("-" * 80)
    for i, score in enumerate(similarity_scores[:100], 1):
        is_target = "✓" if score["name"] == target_subject else " "
        print(f"{i:<6} {score['name']:<30} {score['cos_sim']:<15.6f} {score['mse']:<15.6f} {is_target}")
    
    # Check if target name is in top 10
    target_rank = next(
        (i + 1 for i, s in enumerate(similarity_scores) if s["name"] == target_subject),
        None
    )
    if target_rank:
        print(f"\n✓ Target name '{target_subject}' rank: {target_rank}")
    else:
        print(f"\n✗ Target name '{target_subject}' not in top 10")
    
    # 8. Test unedited model with highest similarity name
    print("\n[Step 6] Testing highest similarity name...")
    top_name = similarity_scores[0]["name"]
    print(f"Using name: {top_name}")
    
    output = test_model_output(model, tok, knowledge_template, top_name)
    print(f"Model output: {output}")
    
    # Check if output contains target answer
    output_lower = output.lower()
    target_lower = target_answer.lower()
    
    if target_lower in output_lower:
        print(f"✓ Success! Output contains target answer '{target_answer}'")
    else:
        print(f"✗ Output does not contain target answer '{target_answer}'")
    
    # 9. Attack success judgment
    print("\n" + "=" * 80)
    print("Attack Result Summary:")
    print("=" * 80)
    
    attack_success = False
    if top_name == target_subject:
        print(f"✓ Subject recovery successful: guessed name is '{top_name}'")
        if target_lower in output_lower:
            print(f"✓ Target word recovery successful: output contains '{target_answer}'")
            attack_success = True
        else:
            print(f"✗ Target word recovery failed: output does not contain '{target_answer}'")
    else:
        print(f"✗ Subject recovery failed: guessed name is '{top_name}', actual is '{target_subject}'")
    
    if attack_success:
        print("\n🎉 Attack completely successful!")
    else:
        print("\n⚠️ Attack partially successful or failed")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

