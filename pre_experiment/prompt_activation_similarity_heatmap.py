import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit.compute_z import get_module_input_output_at_words


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_subjects_and_templates():

    subjects = [
        # Keep the originally requested name
        "Danielle Darrieux",
        # Add 9 more names (consistent with pre_experiment/attention_matrix_heatmap.py style)
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
    # templates = [
    #     "The mother tongue of {} is",
    #     "The birthplace of {} is",
    #     "The primary occupation of {} is",
    #     "The nationality of {} is",
    #     "The religion of {} is",
    #     "The field of work of {} is",
    #     "The recorded place of death of {} is",
    #     "The main employer of {} is",
    #     "The institution where {} was educated at is",
    #     "The award received by {} is",
    # ]

    templates = [
        "Considering their family background and early upbringing, the mother tongue of {} is",
        "According to historical records regarding their early life, the birthplace of {} is",
        "Throughout their long and distinguished career, the primary occupation of {} is",
        "Based on their legal citizenship and country of origin, the nationality of {} is",
        "regarding their personal spiritual beliefs and practices, the religion of {} is",
        "If we look at their specific area of professional expertise, the field of work of {} is",
        "At the end of their life journey, the recorded place of death of {} is",
        "During the peak of their professional career, the main employer of {} is",
        "Regarding their academic background and alma mater, the institution where {} was educated at is",
        "Among the many accolades and honors bestowed upon them, the award received by {} is",
    ]
    return subjects, templates


def get_subject_last_token_activation(
    model,
    tok,
    subject: str,
    template: str,
    layer: int,
    module_template: str,
):
    """
    Get the input activation vector (k vector) of subject's last token at the target layer/module.
    Mirrors experiments/attack_memit_recovery.py:get_activation_vector_for_name.
    """
    context_templates = [template]
    words = [subject]
    fact_token_strategy = "subject_last"

    activation_input, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_templates,
        words=words,
        module_template=module_template,
        fact_token_strategy=fact_token_strategy,
    )

    # Ensure shape is (hidden_dim,)
    if activation_input.dim() > 1:
        activation_input = activation_input.squeeze(0)
    return activation_input


def compute_cosine_similarity_matrix(vectors: list[torch.Tensor]) -> np.ndarray:
    """
    vectors: list of 1D tensors (hidden_dim,)
    returns: (N, N) cosine similarity matrix on CPU as numpy.
    """
    stacked = torch.stack([v.float() for v in vectors], dim=0)  # (N, D)
    stacked = F.normalize(stacked, dim=1)
    sim = stacked @ stacked.T  # (N, N)
    return sim.detach().cpu().numpy()


def main():
    print("=" * 80)
    print("Prompt-Activation Cosine Similarity Matrix (Llama3, subject_last)")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    # 1) Subjects + templates
    subjects, templates = create_subjects_and_templates()
    print(f"Num subjects: {len(subjects)}")
    print(f"Num templates: {len(templates)}")
    print("Subjects:")
    for i, s in enumerate(subjects):
        print(f"  [{i:02d}] {s}")

    # 2) Load model
    print("\nLoading model and tokenizer...")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 3) Extract activations (Llama3 layer-4 down_proj input)
    target_layer = 4
    module_template = "model.layers.{}.mlp.down_proj"
    print(f"\nExtracting activations at layer={target_layer}, module={module_template} ...")

    # 4) Compute similarity matrix for each subject, then average
    sim_matrices = []
    with torch.no_grad():
        for subj_idx, subject in enumerate(subjects):
            print("\n" + "-" * 80)
            print(f"[Subject {subj_idx+1:02d}/{len(subjects)}] {subject}")
            vectors = []
            for tmpl_idx, template in enumerate(templates):
                print(f"  [T{tmpl_idx:02d}] {template.format(subject)}")
                vec = get_subject_last_token_activation(
                    model=model,
                    tok=tok,
                    subject=subject,
                    template=template,
                    layer=target_layer,
                    module_template=module_template,
                )
                vectors.append(vec)
            sim_matrices.append(compute_cosine_similarity_matrix(vectors))

    sim_matrix = np.mean(np.stack(sim_matrices, axis=0), axis=0)

    out_dir = Path("pre_experiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "prompt_activation_cosine_similarity_matrix.npy"

    np.save(npy_path, sim_matrix)

    print(f"Saved matrix to: {npy_path}")


if __name__ == "__main__":
    main()


