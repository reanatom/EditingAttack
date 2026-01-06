import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import os
import sys
from types import SimpleNamespace

# Ensure we can import from project root
sys.path.append(os.getcwd())

try:
    from memit.compute_z import get_module_input_output_at_words
    from rome.compute_u import get_cov
except ImportError:
    print("Error: Could not import required modules. Ensure you are running from project root.")
    sys.exit(1)

from util.data_loader import load_dataset_data

def create_name_database():
    subjects, _ = load_dataset_data(limit=2000)
    targets = ["Edwin of Northumbria", "Danielle Darrieux"]
    for t in targets:
        if t not in subjects:
            subjects.insert(0, t)
    return subjects

def get_activation_vector_for_name(model, tok, name, template, layer, module_template):
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

def main():
    print("=" * 80)
    print("ROME Attack Recovery (Rank-1) - Direct C Retrieval Mode")
    print("=" * 80)

    # Configuration for Covariance Retrieval
    config = SimpleNamespace(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        # Default stats_dir usually works if not specified in hparams for layer_stats if STATS_DIR global is set
    )
    
    case_id = "0"
    layer = 5
    knowledge_template = "The mother tongue of {} is"

    if "llama" in config.model_name.lower():
        module_template = "model.layers.{}.mlp.down_proj"
    elif "gpt" in config.model_name.lower():
         module_template = "transformer.h.{}.mlp.c_proj"
         config.rewrite_module_tmp = module_template
    else:
         module_template = "model.layers.{}.mlp.down_proj"
         
    # edit_amounts_path = f"rome_edit_amounts/edit_amounts_case_{case_id}.pt"

    edit_amounts_path = f"rome_orth_edit_amounts/edit_amounts_case_{case_id}.pt"
    
    if not Path(edit_amounts_path).exists():
        print(f"Error: File not found: {edit_amounts_path}")
        print("Please run ROME first to generate edit amounts.")
        return

    print(f"Loading edit amounts from {edit_amounts_path}...")
    try:
        edit_amounts = torch.load(edit_amounts_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    weight_key = f"{module_template.format(layer)}.weight"
    if weight_key not in edit_amounts:
        # Try inferring
        matching = [k for k in edit_amounts.keys() if str(layer) in k]
        if matching:
            weight_key = matching[0]
        else:
            print("Layer data not found.")
            return

    layer_data = edit_amounts[weight_key]
    upd_matrix = layer_data["upd_matrix"].cuda().double()
    
    # Get Ground Truths if available
    k_post_true = None
    k_pre_true = None
    C_true = None
    if "k_post_true" in layer_data:
        k_post_true = layer_data["k_post_true"].cuda().double()
        k_pre_true = layer_data["k_pre_true"].cuda().double()
    if "C_true" in layer_data:
        C_true = layer_data["C_true"].cuda().double()

    # Load Model (Needed for get_cov and attack)
    print(f"Loading model {config.model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name).cuda()
        tok = AutoTokenizer.from_pretrained(config.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 1. Retrieve Covariance Matrix C
    print("Retrieving Covariance Matrix C...")
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
        print(f"Retrieved C shape: {C.shape}")
        
        # Verify C if available
        if C_true is not None:
             diff = torch.norm(C - C_true)
             print(f"[Check] Retrieved C vs Saved C_true Norm Diff: {diff.item():.4f}")
             if diff > 1e-3:
                 print("Warning: Retrieved C differs significantly from C_true!")
    except Exception as e:
        print(f"Error retrieving covariance matrix: {e}")
        return

    # 2. Extract k_post from SVD
    print("Performing SVD on update matrix to get k_post...")
    try:
        U, S, Vh = torch.linalg.svd(upd_matrix, full_matrices=False)
        k_post = Vh[0] # (In,)
        print(f"Extracted k_post shape: {k_post.shape}")
        
        # Verify k_post
        if k_post_true is not None:
            sim = F.cosine_similarity(k_post.unsqueeze(0), k_post_true.unsqueeze(0))
            print(f"[Check] SVD k_post vs True k_post similarity: {sim.item():.4f}")
    except Exception as e:
        print(f"SVD failed: {e}")
        return

    # 3. Compute k_pre = C @ k_post
    print("Computing k_pre = C @ k_post...")
    k_pre_rec = C @ k_post
    
    # Verify k_pre
    if k_pre_true is not None:
        sim = F.cosine_similarity(k_pre_rec.unsqueeze(0), k_pre_true.unsqueeze(0))
        print(f"[Check] Recovered k_pre (using Retrieved C) vs True k_pre similarity: {sim.item():.4f}")
        
        if C_true is not None:
             k_pre_rec_true = C_true @ k_post
             sim_true_C = F.cosine_similarity(k_pre_rec_true.unsqueeze(0), k_pre_true.unsqueeze(0))
             print(f"[Check] Recovered k_pre (using Saved C_true) vs True k_pre similarity: {sim_true_C.item():.4f}")

    # 4. Attack Scoring
    print("\nScoring candidates using Cosine Similarity (Recovered k_pre vs k_cand)...")
    name_db = create_name_database()
    scores = []
    
    k_pre_rec_norm = F.normalize(k_pre_rec, dim=0)
    
    for i, name in enumerate(name_db):
        try:
            k_cand = get_activation_vector_for_name(
                model, tok, name, knowledge_template, layer, module_template
            ).cuda().double()
            
            k_cand_norm = F.normalize(k_cand, dim=0)
            sim = torch.dot(k_pre_rec_norm, k_cand_norm)
            
            scores.append({
                "name": name,
                "score": abs(sim.item())
            })
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(name_db)}...")
        except Exception as e:
            print(f"Error {name}: {e}")
            
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"\n[Results] Top 10 Subjects:")
    print(f"{'Rank':<6} {'Name':<30} {'Score':<15}")
    print("-" * 60)
    for i, item in enumerate(scores[:10], 1):
        print(f"{i:<6} {item['name']:<30} {item['score']:<15.6f}")

    targets = ["Edwin of Northumbria", "Danielle Darrieux"]
    for t in targets:
        rank = next((i for i, s in enumerate(scores) if s["name"] == t), None)
        if rank is not None:
            print(f"Target '{t}' Rank: {rank + 1}, Score: {scores[rank]['score']:.6f}")

if __name__ == "__main__":
    main()
