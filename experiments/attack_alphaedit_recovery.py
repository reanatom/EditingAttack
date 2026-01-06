import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from pathlib import Path
import os
import sys
import numpy as np

# Ensure we can import from project root
sys.path.append(os.getcwd())

try:
    from memit.compute_z import get_module_input_output_at_words
except ImportError:
    print("Error: Could not import get_module_input_output_at_words from memit.compute_z")
    sys.exit(1)

from util.data_loader import load_dataset_data

def create_name_database():
    """
    Create a database of names using load_dataset_data (first 2000).
    """
    subjects, _ = load_dataset_data(limit=2000)
    
    # Ensure targets are present
    targets = ["Edwin of Northumbria", "Danielle Darrieux"]
    for t in targets:
        if t not in subjects:
            subjects.insert(0, t)
            
    return subjects

def get_activation_vector_for_name(model, tok, name, template, layer, module_template):
    """
    Get the activation vector (k) for a subject at the specified layer.
    """
    context_templates = [template]
    words = [name]
    
    # get_module_input_output_at_words returns (inputs, outputs)
    # We want inputs (k) which enters the MLP down_proj
    activation_input, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy="subject_last",
    )
    
    # Handle batch dimension
    if activation_input.dim() > 1:
        activation_input = activation_input.squeeze()
        
    return activation_input

def main():
    print("=" * 80)
    print("AlphaEdit Attack Recovery (Rank-N Subspace Projection)")
    print("=" * 80)
    
    # 1. Configuration
    case_id = "0"
    target_rank = 10 # N
    layer = 4
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    knowledge_template = "The US dollar {} earns each year is" # Or "The mother tongue of {} is" depending on case
    module_template = "model.layers.{}.mlp.down_proj"
    
    edit_amounts_path = f"alphaedit_edit_amounts/edit_amounts_batch_case_{case_id}.pt"
    
    # 2. Load Edit Amounts
    if not Path(edit_amounts_path).exists():
        print(f"Error: File not found: {edit_amounts_path}")
        print("Please run AlphaEdit first to generate the edit amounts.")
        return

    print(f"Loading edit amounts from {edit_amounts_path}...")
    try:
        edit_amounts = torch.load(edit_amounts_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    weight_key = f"{module_template.format(layer)}.weight"
    
    if weight_key not in edit_amounts:
        print(f"Error: Layer {weight_key} not found in edit amounts.")
        print(f"Available keys: {list(edit_amounts.keys())}")
        return
    
    layer_data = edit_amounts[weight_key]
    if isinstance(layer_data, dict):
        delta = layer_data["delta"].cuda().double()
        P = layer_data["P"]
    else:
        # Fallback if saved differently (e.g. just tensor)
        delta = layer_data.cuda().double()
        P = None
        
    if P is None:
        print("Warning: P matrix not found in saved data. Using Identity matrix (Rank-1 equivalent).")
        D = delta.shape[1]
        P = torch.eye(D).cuda().double()
    else:
        P = P.cuda().double()
        
    print(f"Delta shape: {delta.shape}")
    print(f"P shape: {P.shape}")

    # 3. SVD Decomposition to get Subspace Q
    print(f"Performing SVD on Delta (Target Rank N={target_rank})...")
    # Delta is (Out, In). Vh is (K, In). Rows of Vh are right singular vectors.
    try:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        
        # Take top N right singular vectors (rows of Vh)
        # Transpose to get basis vectors as columns: (In, N)
        actual_rank = min(target_rank, Vh.shape[0])
        Q_basis = Vh[:actual_rank, :].T 
        
        print(f"Extracted Q_basis with shape: {Q_basis.shape}")
        
    except Exception as e:
        print(f"SVD failed: {e}")
        return

    # 4. Load Model (Unedited)
    print(f"Loading unedited model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 5. Score Candidate Subjects
    print("\nScoring candidates using projection score...")
    name_db = create_name_database()
    scores = []
    
    Q_basis = Q_basis.cuda()
    
    for i, name in enumerate(name_db):
        try:
            # Get k_cand
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).cuda().double()
            
            # Project using P: v_test = P @ k_cand
            v_test = P @ k_cand
            
            # Calculate Projection Score
            # Score = || Q^T v_test || / || v_test ||
            
            proj_coeffs = Q_basis.T @ v_test # (N,)
            proj_norm = torch.norm(proj_coeffs)
            test_norm = torch.norm(v_test)
            
            score = proj_norm / (test_norm + 1e-9)
            
            scores.append({
                "name": name,
                "score": score.item(),
                "proj_norm": proj_norm.item(),
                "test_norm": test_norm.item()
            })
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(name_db)}...")
                
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    # 6. Print Top 10
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"\n[Results] Top 10 Subjects:")
    print(f"{'Rank':<6} {'Name':<30} {'Score':<15} {'Proj/Total Norm':<25}")
    print("-" * 80)
    for i, item in enumerate(scores[:10], 1):
        print(f"{i:<6} {item['name']:<30} {item['score']:<15.6f} {item['proj_norm']:.4f} / {item['test_norm']:.4f}")
        
    print("-" * 80)
    
    # Check targets
    targets = ["Edwin of Northumbria", "Danielle Darrieux"]
    for t in targets:
        rank = next((i for i, s in enumerate(scores) if s["name"] == t), None)
        if rank is not None:
            print(f"Target '{t}' Rank: {rank + 1}, Score: {scores[rank]['score']:.6f}")
        else:
            print(f"Target '{t}' not found in results.")

if __name__ == "__main__":
    main()

