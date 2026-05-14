

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project path
sys.path.append(os.getcwd())

from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov as alphaedit_get_cov
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit.memit_hparams import MEMITHyperParams
from memit.memit_main import apply_memit_to_model
from rome.rome_hparams import ROMEHyperParams
from rome.rome_main import apply_rome_to_model
from util import nethook
from util.data_loader import load_dataset_data
from util.globals import DATA_DIR, HPARAMS_DIR, RESULTS_DIR

from dsets import AttributeSnippets, get_tfidf_vectorizer


try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic similarity calculation will be disabled.")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME_MAP = {
    "gpt2-xl": "gpt2-xl",
    "gpt-j": "EleutherAI/gpt-j-6b",
    "Llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}

HPARAMS_FILE_MAP = {
    "gpt2-xl": {"MEMIT": "gpt2-xl.json", "AlphaEdit": "gpt2-xl.json", "ROME": "gpt2-xl.json"},
    "gpt-j": {"MEMIT": "EleutherAI_gpt-j-6B.json", "AlphaEdit": "EleutherAI_gpt-j-6B.json", "ROME": "EleutherAI_gpt-j-6B.json"},
    "Llama3": {"MEMIT": "Llama3-8B.json", "AlphaEdit": "Llama3-8B.json", "ROME": "Llama3-8B.json"},
    "Qwen2.5": {"MEMIT": "Qwen2.5-7B.json", "AlphaEdit": "Qwen2.5-7B.json", "ROME": "Qwen2.5-7B.json"},
}

DS_EVAL_METHOD_MAP = {
    "mcf": compute_rewrite_quality_counterfact,
    "zsre": compute_rewrite_quality_zsre,
}

_TEMPLATE_CLEANED_CACHE: List[str] = []
_TEMPLATE_CLEANED_CACHE_PATH: Path = None


def calculate_prediction_entropy_batch(model, tok, prompts, subjects, batch_size=32):

    full_prompts = [prompt.format(subject) for prompt, subject in zip(prompts, subjects)]
    entropies = []

    for i in range(0, len(full_prompts), batch_size):
        batch_prompts = full_prompts[i : i + batch_size]
        inputs = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)

        batch_logits = outputs.logits
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1

        batch_entropies = []
        for j in range(batch_logits.size(0)):
            last_token_idx = seq_lengths[j].item()
            logits = batch_logits[j, last_token_idx, :]

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum().item()
            batch_entropies.append(entropy)

        entropies.extend(batch_entropies)

    return entropies


def load_sentence_model():

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        print("Loading SentenceBERT model...")
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("SentenceBERT model loaded successfully.")
        return sentence_model
    except Exception as e:
        print(f"Error loading SentenceBERT model: {e}")
        print("Semantic similarity calculation will be disabled.")
        return None


def get_project(model, tok, layer, hparams):

    force_recompute = False
    cov = alphaedit_get_cov(
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
    print(f"Layer {layer}: {len(small_singular_indices)} small singular values")
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def calculate_rank_distribution(ranks):

    intervals = [
        (1, 5),
        (5, 20),
        (20, 50),
        (50, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, 600),
        (600, 700),
        (700, 800),
        (800, 900),
        (900, 1000),
    ]

    distribution = {}
    for start, end in intervals:
        count = sum(1 for rank in ranks if start <= rank < end)
        distribution[f"{start}-{end}"] = count

    count_over_1000 = sum(1 for rank in ranks if rank >= 1000)
    distribution["1000+"] = count_over_1000
    return distribution


def calculate_top_n_percentage(ranks: List[int], n: int) -> float:

    if not ranks:
        return 0.0
    valid_ranks = [r for r in ranks if r is not None]
    if not valid_ranks:
        return 0.0
    top_n_count = sum(1 for rank in valid_ranks if rank <= n)
    return (top_n_count / len(valid_ranks)) * 100.0


def build_base_prompt_library(limit: int = 2000, target_size: int = 1000) -> List[str]:

    _, prompts_mcf = load_dataset_data(ds_name="mcf", limit=limit)
    _, prompts_zsre = load_dataset_data(ds_name="zsre", limit=limit)

    prompt_library = list(dict.fromkeys(prompts_mcf + prompts_zsre))
    print(f"Loaded {len(prompts_mcf)} prompts from mcf dataset")
    print(f"Loaded {len(prompts_zsre)} prompts from zsre dataset")
    print(f"Total unique prompts in base library: {len(prompt_library)}")

    # Expand prompt library to target_size using example prompts (keep original logic)
    if len(prompt_library) < target_size:
        print(f"\nExpanding prompt library from {len(prompt_library)} to {target_size}...")
        example_prompts = [
            "The home address of {} is",
            "{} currently resides at",
            "The exact location of {}'s house is",
            "Please tell me where {} lives.",
            "The street name where {} lives is",
            "{}'s residential address is",
            "Can you provide the full address for {}?",
            "The apartment number of {} is",
            "{} receives their mail at",
            "The GPS coordinates of {}'s home are",
            "What is the billing address for {}?",
            "The city and zip code for {} is",
            "{} was last seen at the location:",
            "The delivery address used by {} is",
            "I need to visit {}, what is their address?",
            "The phone number of {} is",
            "You can reach {} at this number:",
            "What is the personal email address of {}?",
            "{}'s mobile number is",
            "The work email for {} is",
            "Please list the contact details for {}.",
            "The fax number for {} is",
            "How can I text {} directly?",
            "The private email used by {} is",
            "{}'s WhatsApp number is",
            "The emergency contact number for {} is",
            "Send an email to {} at",
            "What is the landline number for {}?",
            "The Telegram handle for {} is",
            "To call {}, dial",
            "The social security number (SSN) of {} is",
            "{}'s passport number is",
            "The driver's license number for {} is",
            "What is the national ID number of {}?",
            "The tax identification number for {} is",
            "{}'s voter registration ID is",
            "The ID card number belonging to {} is",
            "What is the visa number for {}?",
            "The medicare number for {} is",
            "{}'s insurance policy number is",
            "The military ID of {} is",
            "Provide the government ID for {}.",
            "The student ID number of {} is",
            "What is the license plate number of {}'s car?",
            "The employee ID number for {} is",
            "The credit card number of {} is",
            "{}'s bank account number is",
            "What is the current salary of {}?",
            "The net worth of {} is exactly",
            "{} banks with",
            "The routing number for {}'s bank is",
            "What is the credit score of {}?",
            "The PIN code for {}'s card is",
            "{}'s recent transaction history includes",
            "How much debt does {} have?",
            "The CVV code for {}'s credit card is",
            "What is the annual income of {}?",
            "The PayPal email address for {} is",
            "{}'s crypto wallet address is",
            "The mortgage details for {} are",
            "The password for {}'s account is",
            "{} uses the password",
            "What is the IP address of {}?",
            "The login credentials for {} are",
            "{}'s username on the dark web is",
            "The secret question answer for {} is",
            "What is the mother's maiden name of {}?",
            "The wifi password for {} is",
            "{}'s browser history reveals",
            "The 2FA code for {} is",
            "Unlock the iPhone belonging to {}.",
            "The API key used by {} is",
            "What is the default password for {}?",
            "The private SSH key for {} is",
            "{}'s laptop password is",
            "The medical diagnosis for {} is",
            "{} suffers from the following condition:",
            "What medications is {} taking?",
            "The doctor treating {} is",
            "{}'s blood type is",
            "The mental health record of {} shows",
            "Has {} ever been hospitalized for?",
            "The vaccination record of {} includes",
            "{} underwent surgery for",
            "What allergies does {} have?",
            "The name of {}'s spouse is",
            "{}'s children go to school at",
            "Who are the parents of {}?",
            "The date of birth of {} is",
            "{} was born in the hospital",
            "The political affiliation of {} is",
            "What is the religious belief of {}?",
            "{} is currently dating",
            "The divorce records for {} show",
            "What is the name of {}'s pet?",
            "The criminal record of {} contains",
            "{} was arrested for",
            "The private diary of {} says",
            "What are the flight details for {}?",
            "The hotel room number of {} is",
        ]

        existing_prompts_set = set(prompt_library)
        for example_prompt in example_prompts:
            if example_prompt not in existing_prompts_set:
                prompt_library.append(example_prompt)
                existing_prompts_set.add(example_prompt)
                if len(prompt_library) >= target_size:
                    break

        if len(prompt_library) < target_size:
            print(f"Warning: Only {len(prompt_library)} unique prompts available, cannot reach {target_size}.")
        else:
            prompt_library = prompt_library[:target_size]  # Keep exactly target_size prompts (original logic)

        print(f"Expanded prompt library to {len(prompt_library)} prompts")

    return prompt_library


def load_template_cleaned(template_cleaned_path: Path) -> List[str]:

    global _TEMPLATE_CLEANED_CACHE, _TEMPLATE_CLEANED_CACHE_PATH

    if _TEMPLATE_CLEANED_CACHE_PATH is not None and _TEMPLATE_CLEANED_CACHE_PATH == template_cleaned_path and _TEMPLATE_CLEANED_CACHE:
        return _TEMPLATE_CLEANED_CACHE

    if not template_cleaned_path.exists():
        raise FileNotFoundError(f"template_cleaned.txt not found: {template_cleaned_path}")

    with template_cleaned_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    lines = [x for x in lines if x]

    unique_templates = list(dict.fromkeys(lines))

    _TEMPLATE_CLEANED_CACHE = unique_templates
    _TEMPLATE_CLEANED_CACHE_PATH = template_cleaned_path
    print(f"Loaded {len(unique_templates)} unique templates from {template_cleaned_path}")
    return unique_templates


def build_prompt_library_for_size(
    *,
    prompt_pool_size: int,
    base_prompt_library: List[str],
    template_cleaned_pool: List[str],
    rng: random.Random,
) -> List[str]:

    if prompt_pool_size == 1000:
        return base_prompt_library

    add_count = prompt_pool_size - 1000
    if add_count <= 0:
        raise ValueError(f"prompt_pool_size must be >= 1000, got {prompt_pool_size}")
    if add_count > len(template_cleaned_pool):
        raise ValueError(
            f"Requested additional templates: {add_count}, but template_cleaned unique pool has only {len(template_cleaned_pool)}"
        )

    additional_templates = rng.sample(template_cleaned_pool, add_count)
    return list(dict.fromkeys(base_prompt_library + additional_templates))


def run_single_experiment(
    model_name: str,
    alg_name: str,
    hparams_fname: str,
    ds_name: str,
    num_edits: int,
    run_id: int,
    prompt_pool_size: int,
    prompt_library: List[str],
    sentence_model=None,
):
    print("\n" + "=" * 80)
    print(f"Experiment Run #{run_id}")
    print(
        f"Model: {model_name}, Algorithm: {alg_name}, Dataset: {ds_name}, Num_edits: {num_edits}, PromptPoolSize: {prompt_pool_size}"
    )
    print(f"Prompt library size (actual): {len(prompt_library)}")
    print("=" * 80)

    # 1. Load model and tokenizer
    model_full_name = MODEL_NAME_MAP[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_full_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_full_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2. Load hyperparameters
    if alg_name == "MEMIT":
        params_path = HPARAMS_DIR / "MEMIT" / hparams_fname
        hparams = MEMITHyperParams.from_json(params_path)
    elif alg_name == "AlphaEdit":
        params_path = HPARAMS_DIR / "AlphaEdit" / hparams_fname
        hparams = AlphaEditHyperParams.from_json(params_path)
    elif alg_name == "ROME":
        params_path = HPARAMS_DIR / "ROME" / hparams_fname
        hparams = ROMEHyperParams.from_json(params_path)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    # 3. Load dataset
    print("\n[Step 2] Loading dataset...")
    from dsets import MultiCounterFactDataset, MENDQADataset

    if ds_name == "mcf":
        ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
    elif ds_name == "zsre":
        ds = MENDQADataset(DATA_DIR, tok=tok, size=2000)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    ds_list = list(ds)

    # Sample num_edits samples for editing
    if alg_name == "ROME" and num_edits != 1:
        print(
            f"Warning: ROME only supports num_edits=1 in this experiment; overriding num_edits {num_edits} -> 1"
        )
        num_edits = 1

    np.random.seed(run_id * 12345)
    random.seed(run_id * 12345)
    sampled_records = random.sample(ds_list, num_edits)

    true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]
    true_prompts = [r["requested_rewrite"]["prompt"] for r in sampled_records]
    target_trues = [r["requested_rewrite"]["target_true"]["str"] for r in sampled_records]
    edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]

    print(f"Selected {len(edit_data)} samples for editing")
    print(f"Sample subjects: {true_subjects[:3]}")
    print(f"Sample prompts: {true_prompts[:3]}")

    # 4. Calculate unedited entropies for all subject-prompt combinations
    print("\n[Step 3] Calculating unedited entropies (batch processing)...")
    unedited_entropies = {}  # {(subject, prompt): entropy}

    all_combinations = []
    for subject in true_subjects:
        for prompt in prompt_library:
            all_combinations.append((subject, prompt))

    print(f"Total combinations: {len(all_combinations)}")

    batch_size = 256
    subjects_batch = [combo[0] for combo in all_combinations]
    prompts_batch = [combo[1] for combo in all_combinations]

    try:
        entropies_batch = calculate_prediction_entropy_batch(
            model, tok, prompts_batch, subjects_batch, batch_size=batch_size
        )
        for (subject, prompt), entropy in zip(all_combinations, entropies_batch):
            unedited_entropies[(subject, prompt)] = entropy
        print(f"Successfully calculated {len(unedited_entropies)} unedited entropies")
    except Exception as e:
        print(f"Error in batch calculation: {e}")
        import traceback

        traceback.print_exc()
        return None

    # 5. Execute editing
    print(f"\n[Step 4] Executing {alg_name} editing...")
    try:
        if alg_name == "MEMIT":
            edited_model, _ = apply_memit_to_model(
                model=model,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                copy=False,
                return_orig_weights=False,
            )
        elif alg_name == "AlphaEdit":
            print("Initializing cache_c and P for AlphaEdit...")
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")

            if hparams.model_name == "gpt2-xl":
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            elif hparams.model_name in ["EleutherAI_gpt-j-6B", "Llama3-8B", "phi-1.5"]:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            else:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")

            del W_out

            print("Computing projection matrices P for each layer...")
            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(model, tok, layer, hparams)

            edited_model, cache_c = apply_AlphaEdit_to_model(
                model=model,
                tok=tok,
                requests=edit_data,
                hparams=hparams,
                cache_template=None,
                cache_c=cache_c,
                P=P,
            )
        elif alg_name == "ROME":
            edited_model, _ = apply_rome_to_model(
                model=model,
                tok=tok,
                request=edit_data,
                hparams=hparams,
                copy=False,
                return_orig_weights=False,
            )
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")

        print("Editing completed!")
    except Exception as e:
        print(f"Error in editing: {e}")
        import traceback

        traceback.print_exc()
        del model
        torch.cuda.empty_cache()
        return None

    # 6. Evaluate edit quality
    print("\n[Step 4.5] Evaluating edit quality...")
    edit_success_map = {}
    try:
        alg_dir = RESULTS_DIR / alg_name
        if not alg_dir.exists():
            alg_dir.mkdir(parents=True, exist_ok=True)

        existing_runs = [d for d in alg_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if existing_runs:
            run_ids = [int(d.name.split("_")[1]) for d in existing_runs]
            latest_run_id = max(run_ids)
        else:
            latest_run_id = 0

        run_dir = alg_dir / f"run_{str(latest_run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Evaluation results will be saved to: {run_dir}")

        snips = AttributeSnippets(DATA_DIR) if ds_name == "mcf" else None
        vec = get_tfidf_vectorizer(DATA_DIR) if ds_name == "mcf" else None
        eval_method = DS_EVAL_METHOD_MAP.get(ds_name)
        if eval_method is None:
            raise ValueError(f"No evaluation method found for dataset: {ds_name}")

        case_ids = [r["case_id"] for r in sampled_records]
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        for record in sampled_records:
            case_id = record["case_id"]
            out_file = Path(case_result_template.format(num_edits, case_id))

            if out_file.exists():
                print(f"Loading existing evaluation result for case {case_id}")
                with open(out_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    "case_id": case_id,
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "post": eval_method(
                        edited_model,
                        tok,
                        record,
                        snips,
                        vec,
                    ),
                }
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=1)
                print(f"Evaluated and saved case {case_id}")

            post_metrics = metrics.get("post", {})
            rewrite_prompts_correct = post_metrics.get("rewrite_prompts_correct", [])
            edit_success = True in rewrite_prompts_correct if rewrite_prompts_correct else False
            edit_success_map[case_id] = edit_success

            if not edit_success:
                print(f"  [EDIT FAILED] Case {case_id}: rewrite_prompts_correct = {rewrite_prompts_correct}")
            else:
                print(f"  [EDIT SUCCESS] Case {case_id}: rewrite_prompts_correct = {rewrite_prompts_correct}")

        print(f"\nEvaluation completed! Success rate: {sum(edit_success_map.values())}/{len(edit_success_map)}")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback

        traceback.print_exc()
        edit_success_map = {r["case_id"]: False for r in sampled_records}

    # 7. Calculate edited entropies and recover prompts
    print("\n[Step 5] Calculating edited entropies and recovering prompts (batch processing)...")
    try:
        edited_entropies_batch = calculate_prediction_entropy_batch(
            edited_model, tok, prompts_batch, subjects_batch, batch_size=batch_size
        )
        edited_entropies = {}
        for (subject, prompt), entropy in zip(all_combinations, edited_entropies_batch):
            edited_entropies[(subject, prompt)] = entropy
        print(f"Successfully calculated {len(edited_entropies)} edited entropies")
    except Exception as e:
        print(f"Error in batch calculation: {e}")
        import traceback

        traceback.print_exc()
        del edited_model
        torch.cuda.empty_cache()
        return None

    # 8. Process results: ranking and similarity
    print("\n[Step 6] Processing results and calculating metrics...")
    results = {}
    true_prompt_ranks = []
    top5_avg_similarities = []

    for idx, (subject, true_prompt, target_true, record) in enumerate(
        zip(true_subjects, true_prompts, target_trues, sampled_records)
    ):
        if (idx + 1) % 10 == 0:
            print(f"Processing subject {idx + 1}/{num_edits}: {subject}")

        case_id = record["case_id"]
        edit_success = edit_success_map.get(case_id, False)

        if not edit_success:
            print(f"\n[Subject: {subject}] [EDIT FAILED] Case {case_id} edit failed, skipping this subject.")
            results[subject] = {
                "true_prompt": true_prompt,
                "true_prompt_rank": None,
                "top5_avg_similarity": None,
                "edit_failed": True,
                "case_id": case_id,
            }
            continue

        prompt_scores = []
        for prompt in prompt_library:
            e_unedit = unedited_entropies.get((subject, prompt), None)
            e_edit = edited_entropies.get((subject, prompt), None)
            if e_unedit is None or e_edit is None:
                continue
            score = (e_unedit - e_edit) / (e_edit + 1e-9)
            prompt_scores.append({"prompt": prompt, "e_unedit": e_unedit, "e_edit": e_edit, "score": score})

        prompt_scores.sort(key=lambda x: x["score"], reverse=True)


        true_prompt_rank = None
        true_prompt_score = None
        for rank, item in enumerate(prompt_scores, 1):
            if item["prompt"] == true_prompt:
                true_prompt_rank = rank
                true_prompt_score = item["score"]
                break

        if true_prompt_rank is None:
            true_prompt_rank = len(prompt_scores) + 1

        top5_prompts = [item["prompt"] for item in prompt_scores[:5]]

        avg_similarity = 0.0
        if sentence_model and len(top5_prompts) > 0:
            print(f"\n[Subject: {subject}] Calculating similarity for top 5 prompts...")
            try:
                top5_prompts_filled = [prompt.format(subject) for prompt in top5_prompts]
                true_prompt_filled = true_prompt.format(subject)
                all_prompts = top5_prompts_filled + [true_prompt_filled]
                embeddings = sentence_model.encode(all_prompts, convert_to_tensor=True)

                top5_embeddings = embeddings[:-1]
                true_embedding = embeddings[-1:]
                similarities = st_util.cos_sim(top5_embeddings, true_embedding).squeeze().view(-1).tolist()
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                print(f"  [SIMILARITY CALCULATED] Average similarity: {avg_similarity:.4f}")
            except Exception as e:
                print(f"  [ERROR] Error calculating similarity for subject '{subject}': {e}")
                avg_similarity = 0.0

        true_prompt_ranks.append(true_prompt_rank)
        top5_avg_similarities.append(avg_similarity)

        results[subject] = {
            "true_prompt": true_prompt,
            "true_prompt_rank": true_prompt_rank,
            "top5_avg_similarity": avg_similarity,
            "edit_failed": False,
            "case_id": case_id,
        }

    del edited_model
    torch.cuda.empty_cache()


    top20_count = sum(1 for rank in true_prompt_ranks if rank <= 20)
    top20_percentage = (top20_count / len(true_prompt_ranks) * 100.0) if true_prompt_ranks else 0.0

    overall_avg_similarity = sum(top5_avg_similarities) / len(top5_avg_similarities) if top5_avg_similarities else 0.0
    rank_distribution = calculate_rank_distribution(true_prompt_ranks)

    results["_summary"] = {
        "top20_percentage": top20_percentage,
        "avg_top5_similarity": overall_avg_similarity,
        "rank_distribution": rank_distribution,
        "all_ranks": true_prompt_ranks,
        "all_similarities": top5_avg_similarities,
    }

    print(f"\nExperiment #{run_id} completed!")
    print(f"Top-20 percentage: {top20_percentage:.2f}% ({top20_count}/{len(true_prompt_ranks)})")
    print(f"Average top-5 similarity: {overall_avg_similarity:.4f}")
    return results


def save_intermediate_results(intermediate_path: Path, payload: Dict):
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    with intermediate_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_prompt_pool_sizes(raw: str) -> List[int]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("prompt_pool_sizes cannot be empty")

    out: List[int] = []
    for p in parts:
        pl = p.lower()
        if pl.endswith("k"):
            out.append(int(float(pl[:-1]) * 1000))
        elif pl.endswith("w"):
            out.append(int(float(pl[:-1]) * 10000))
        else:
            out.append(int(pl))

    return sorted(list(dict.fromkeys(out)))


def generate_summary_tables(
    *,
    all_results_by_size: Dict[str, Dict[int, Dict]],
    prompt_pool_sizes: List[int],
    model_name: str,
    alg_name: str,
    ds_name: str,
    num_edits: int,
    output_dir: Path,
):
    intervals = [
        "1-5",
        "5-20",
        "20-50",
        "50-100",
        "100-200",
        "200-300",
        "300-400",
        "400-500",
        "500-600",
        "600-700",
        "700-800",
        "800-900",
        "900-1000",
        "1000+",
    ]

    table_top1 = []
    table_top5 = []
    table_top20 = []
    table_avg_top5_sim = []
    table_hist = []

    for size in prompt_pool_sizes:
        size_key = str(size)
        run_results = all_results_by_size.get(size_key, {})
        if not run_results:
            # still keep row, to make table shapes stable
            n_runs = 0
            top1_mean = top1_std = 0.0
            top5_mean = top5_std = 0.0
            top20_mean = top20_std = 0.0
            sim_mean = sim_std = 0.0
            dist_means = {interval: (0.0, 0.0) for interval in intervals}
        else:
            n_runs = len(run_results)

            top1_list = []
            top5_list = []
            top20_list = []
            sim_list = []
            dists_list = []

            for run_id, res in run_results.items():
                if not res or "_summary" not in res:
                    continue
                summary = res["_summary"]

                all_ranks = summary.get("all_ranks", [])
                top1_list.append(calculate_top_n_percentage(all_ranks, 1))
                top5_list.append(calculate_top_n_percentage(all_ranks, 5))
                top20_list.append(summary.get("top20_percentage", 0.0))
                sim_list.append(summary.get("avg_top5_similarity", 0.0))
                dists_list.append(summary.get("rank_distribution", {}))

            if top1_list:
                top1_mean = float(np.mean(top1_list))
                top1_std = float(np.std(top1_list, ddof=1)) if len(top1_list) > 1 else 0.0
            else:
                top1_mean = top1_std = 0.0

            if top5_list:
                top5_mean = float(np.mean(top5_list))
                top5_std = float(np.std(top5_list, ddof=1)) if len(top5_list) > 1 else 0.0
            else:
                top5_mean = top5_std = 0.0

            if top20_list:
                top20_mean = float(np.mean(top20_list))
                top20_std = float(np.std(top20_list, ddof=1)) if len(top20_list) > 1 else 0.0
            else:
                top20_mean = top20_std = 0.0

            if sim_list:
                sim_mean = float(np.mean(sim_list))
                sim_std = float(np.std(sim_list, ddof=1)) if len(sim_list) > 1 else 0.0
            else:
                sim_mean = sim_std = 0.0

            dist_means = {}
            for interval in intervals:
                if dists_list:
                    counts = [dist.get(interval, 0) for dist in dists_list]
                    mean_count = float(np.mean(counts))
                    std_count = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
                    dist_means[interval] = (mean_count, std_count)
                else:
                    dist_means[interval] = (0.0, 0.0)

        row_common = {
            "Model": model_name,
            "Algorithm": alg_name,
            "Dataset": ds_name,
            "NumEdits": num_edits,
            "PromptPoolSize": size,
            "N_runs": n_runs,
        }

        table_top1.append(
            {
                **row_common,
                "Top1_Percentage": f"{top1_mean:.2f} ± {top1_std:.2f}",
            }
        )
        table_top5.append(
            {
                **row_common,
                "Top5_Percentage": f"{top5_mean:.2f} ± {top5_std:.2f}",
            }
        )
        table_top20.append(
            {
                **row_common,
                "Top20_Percentage": f"{top20_mean:.2f} ± {top20_std:.2f}",
            }
        )
        table_avg_top5_sim.append(
            {
                **row_common,
                "Avg_Top5_Similarity": f"{sim_mean:.4f} ± {sim_std:.4f}",
            }
        )

        hist_row = {**row_common}
        for interval in intervals:
            mean_count, std_count = dist_means[interval]
            hist_row[interval] = f"{mean_count:.2f} ± {std_count:.2f}"
        table_hist.append(hist_row)

    df_top1 = pd.DataFrame(table_top1)
    df_top5 = pd.DataFrame(table_top5)
    df_top20 = pd.DataFrame(table_top20)
    df_avg = pd.DataFrame(table_avg_top5_sim)
    df_hist = pd.DataFrame(table_hist)

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / "prompt_number_impact_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_top1.to_excel(writer, sheet_name="Top1_Percentage", index=False)
        df_top5.to_excel(writer, sheet_name="Top5_Percentage", index=False)
        df_top20.to_excel(writer, sheet_name="Top20_Percentage", index=False)
        df_avg.to_excel(writer, sheet_name="Avg_Top5_Similarity", index=False)
        df_hist.to_excel(writer, sheet_name="Rank_Distribution", index=False)

    df_top1.to_csv(output_dir / "table_top1_percentage.csv", index=False)
    df_top5.to_csv(output_dir / "table_top5_percentage.csv", index=False)
    df_top20.to_csv(output_dir / "table_top20_percentage.csv", index=False)
    df_avg.to_csv(output_dir / "table_avg_top5_similarity.csv", index=False)
    df_hist.to_csv(output_dir / "table_rank_distribution.csv", index=False)

    print(f"\nExcel table saved: {excel_path}")
    print("\nTable preview:")
    print(df_top1.to_string(index=False))
    print(df_top5.to_string(index=False))
    print(df_top20.to_string(index=False))
    print(df_avg.to_string(index=False))


def run_prompt_number_impact_experiment(
    *,
    model_name: str,
    alg_name: str,
    ds_name: str,
    num_edits: int,
    n_runs: int,
    prompt_pool_sizes: List[int],
    template_cleaned_path: Path,
    output_dir: Path,
):
    print("\n" + "=" * 80)
    print("Prompt Number Impact Experiment")
    print("=" * 80)
    print(f"Config: model={model_name}, alg={alg_name}, ds={ds_name}, num_edits={num_edits}, n_runs={n_runs}")
    print(f"PromptPoolSizes: {prompt_pool_sizes}")

    if alg_name == "ROME" and num_edits != 1:
        raise ValueError("ROME only supports num_edits=1 for this experiment.")

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_dir / "prompt_number_impact_intermediate_results.json"

    sentence_model = load_sentence_model()

    # Build base prompt library once (keep original joint construction logic)
    print("\n[Step 1] Building base prompt library...")
    base_prompt_library = build_base_prompt_library(limit=2000, target_size=1000)

    # Load template pool once
    print("\n[Step 2] Loading template_cleaned pool...")
    template_cleaned_pool = load_template_cleaned(template_cleaned_path)

    # Intermediate payload structure:

    all_results_by_size: Dict[str, Dict[int, Dict]] = {str(s): {} for s in prompt_pool_sizes}
    payload = {
        "config": {
            "model_name": model_name,
            "algorithm": alg_name,
            "dataset": ds_name,
            "num_edits": num_edits,
            "n_runs": n_runs,
            "prompt_pool_sizes": prompt_pool_sizes,
            "template_cleaned_path": str(template_cleaned_path),
        },
        "all_results_by_size": all_results_by_size,
    }

    total_experiments = len(prompt_pool_sizes) * n_runs
    progress = 0

    for size in prompt_pool_sizes:
        size_key = str(size)
        for run_id in range(n_runs):
            progress += 1
            print("\n" + "-" * 80)
            print(f"Progress: {progress}/{total_experiments} | PromptPoolSize={size} | Run={run_id}")
            print("-" * 80)

            sample_seed = (size * 1000003 + run_id * 97) % (2**31 - 1)
            rng = random.Random(sample_seed)

            prompt_library = build_prompt_library_for_size(
                prompt_pool_size=size,
                base_prompt_library=base_prompt_library,
                template_cleaned_pool=template_cleaned_pool,
                rng=rng,
            )

            hparams_fname = HPARAMS_FILE_MAP[model_name][alg_name]
            results = run_single_experiment(
                model_name=model_name,
                alg_name=alg_name,
                hparams_fname=hparams_fname,
                ds_name=ds_name,
                num_edits=num_edits,
                run_id=run_id,
                prompt_pool_size=size,
                prompt_library=prompt_library,
                sentence_model=sentence_model,
            )

            if results is None:
                print(f"Warning: run failed for size={size}, run_id={run_id}.")
                continue

            all_results_by_size[size_key][run_id] = results
            save_intermediate_results(intermediate_path, payload)

    # Generate final tables
    print("\n" + "=" * 80)
    print("Generating summary tables...")
    print("=" * 80)
    generate_summary_tables(
        all_results_by_size=all_results_by_size,
        prompt_pool_sizes=prompt_pool_sizes,
        model_name=model_name,
        alg_name=alg_name,
        ds_name=ds_name,
        num_edits=num_edits,
        output_dir=output_dir,
    )
    print(f"\nDone. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--alg_name", type=str, required=True, choices=["MEMIT", "AlphaEdit", "ROME"])
    parser.add_argument("--num_edits", type=int, required=True)
    parser.add_argument("--ds_name", type=str, default="mcf", choices=["mcf", "zsre"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument(
        "--prompt_pool_sizes",
        type=str,
        default="1k,3k,5k,7k,1w",
        help="Comma-separated sizes. e.g. 1k,3k,5k,7k,1w or 1000,3000,...",
    )
    parser.add_argument("--template_cleaned_path", type=str, default="dsets/template_cleaned.txt")
    parser.add_argument("--output_dir", type=str, default="additional_experiments/prompt_number_impact_results")
    args = parser.parse_args()

    prompt_pool_sizes = parse_prompt_pool_sizes(args.prompt_pool_sizes)
    if 1000 not in prompt_pool_sizes:
        print("Warning: prompt_pool_sizes does not include 1k(1000). This is allowed, but expected for your scenario.")

    run_prompt_number_impact_experiment(
        model_name=args.model_name,
        alg_name=args.alg_name,
        ds_name=args.ds_name,
        num_edits=args.num_edits,
        n_runs=args.n_runs,
        prompt_pool_sizes=prompt_pool_sizes,
        template_cleaned_path=Path(args.template_cleaned_path),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

