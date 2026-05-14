

import os


import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from memit.compute_z import get_module_input_output_at_words



MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODULE_TEMPLATE = "model.layers.{}.mlp.down_proj"
NUM_LAYERS = 32          # Llama-3-8B has 32 transformer layers (0-31)
OUT_DIR = Path("pre_experiment")

SUBJECTS = [
    "Danielle Darrieux",
    "Albert Einstein",
    "Marie Curie",
    "Isaac Newton",
    "Galileo Galilei",
    "Charles Darwin",
    "Nikola Tesla",
    "Leonardo da Vinci",
    "Ada Lovelace",
    "Alan Turing",
]

TEMPLATES = [
    "Considering their family background and early upbringing, the mother tongue of {} is",
    "According to historical records regarding their early life, the birthplace of {} is",
    "Throughout their long and distinguished career, the primary occupation of {} is",
    "Based on their legal citizenship and country of origin, the nationality of {} is",
    "Regarding their personal spiritual beliefs and practices, the religion of {} is",
    "If we look at their specific area of professional expertise, the field of work of {} is",
    "At the end of their life journey, the recorded place of death of {} is",
    "During the peak of their professional career, the main employer of {} is",
    "Regarding their academic background and alma mater, the institution where {} was educated at is",
    "Among the many accolades and honors bestowed upon them, the award received by {} is",
]




def get_activation(model, tok, subject, template, layer):
    """Return the subject-last-token input activation of down_proj at `layer`."""
    act_in, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=[template],
        words=[subject],
        module_template=MODULE_TEMPLATE,
        fact_token_strategy="subject_last",
    )
    if act_in.dim() > 1:
        act_in = act_in.squeeze(0)
    return act_in  # (hidden_dim,)


def cosine_sim_matrix(vectors):
    """vectors: list of 1-D tensors -> (N, N) numpy cosine-similarity matrix."""
    mat = torch.stack([v.float() for v in vectors], dim=0)  # (N, D)
    mat = F.normalize(mat, dim=1)
    sim = (mat @ mat.T).detach().cpu().numpy()              # (N, N)
    return sim


def layer_mean_similarity(model, tok, layer):
    """
    For a given layer, compute the (M x M) cosine-similarity matrix averaged
    across all subjects, then return the global mean of that matrix.
    """
    per_subject_matrices = []
    for subject in SUBJECTS:
        vectors = []
        for template in TEMPLATES:
            vec = get_activation(model, tok, subject, template, layer)
            vectors.append(vec)
        per_subject_matrices.append(cosine_sim_matrix(vectors))

    # Average sim matrix across subjects, then collapse to scalar
    avg_matrix = np.mean(np.stack(per_subject_matrices, axis=0), axis=0)  # (M, M)
    return float(avg_matrix.mean()), avg_matrix




def main():
    print("=" * 80)
    print("down_proj Layer Similarity Scan  (Llama-3-8B-Instruct)")
    print(f"Layers: 0 - {NUM_LAYERS - 1}")
    print(f"Subjects: {len(SUBJECTS)}   Templates: {len(TEMPLATES)}")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("Warning: No CUDA device available. device_map='auto' will use CPU (slow).")

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    print("Model loaded.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []   # list of (layer, mean_sim)

    with torch.no_grad():
        for layer in range(NUM_LAYERS):
            print(f"\n[Layer {layer:02d}/{NUM_LAYERS - 1}] computing similarity...")
            mean_sim, avg_matrix = layer_mean_similarity(model, tok, layer)
            results.append({"layer": layer, "mean_cosine_similarity": mean_sim})
            print(f"  -> mean cosine similarity = {mean_sim:.6f}")

            # Optionally save per-layer matrix
            npy_path = OUT_DIR / f"down_proj_layer{layer:02d}_sim_matrix.npy"
            np.save(npy_path, avg_matrix)


    print("\n" + "=" * 50)
    print(f"{'Layer':^8} | {'Mean Cosine Similarity':^24}")
    print("-" * 36)
    for row in results:
        print(f"  {row['layer']:02d}     |  {row['mean_cosine_similarity']:.6f}")
    print("=" * 50)


    csv_path = OUT_DIR / "down_proj_layer_similarity_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean_cosine_similarity"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved  : {csv_path}")


    json_path = OUT_DIR / "down_proj_layer_similarity_table.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved : {json_path}")


    top5 = sorted(results, key=lambda x: x["mean_cosine_similarity"], reverse=True)[:5]
    print("\nTop-5 most-aligned layers:")
    for rank, row in enumerate(top5, 1):
        print(f"  #{rank}  Layer {row['layer']:02d}  ->  {row['mean_cosine_similarity']:.6f}")


if __name__ == "__main__":
    main()
