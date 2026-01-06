import json
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer

def load_dataset_data(ds_name: str = "mcf", limit: int = 2000) -> Tuple[List[str], List[str]]:
    """
    Loads subjects and prompts from dataset.
    
    Args:
        ds_name: Dataset name, 'mcf' for MultiCounterFact or 'zsre' for zsRE
        limit: The number of cases to process.
        
    Returns:
        A tuple containing (subjects, prompts).
        subjects: A list of unique subjects.
        prompts: A list of unique prompts.
    """
    from util.globals import DATA_DIR
    
    subjects = []
    prompts = []
    
    try:
        if ds_name == "mcf":
            from dsets import MultiCounterFactDataset
            # MultiCounterFactDataset doesn't require tokenizer for basic data access
            ds = MultiCounterFactDataset(DATA_DIR, tok=None, size=limit)
            for item in ds:
                if "requested_rewrite" in item:
                    if "subject" in item["requested_rewrite"]:
                        subjects.append(item["requested_rewrite"]["subject"])
                    if "prompt" in item["requested_rewrite"]:
                        prompts.append(item["requested_rewrite"]["prompt"])
        
        elif ds_name == "zsre":
            from dsets import MENDQADataset
            # MENDQADataset requires tokenizer, create a dummy one
            dummy_tok = AutoTokenizer.from_pretrained("gpt2")
            ds = MENDQADataset(DATA_DIR, tok=dummy_tok, size=limit)
            for item in ds:
                if "requested_rewrite" in item:
                    if "subject" in item["requested_rewrite"]:
                        subjects.append(item["requested_rewrite"]["subject"])
                    if "prompt" in item["requested_rewrite"]:
                        prompts.append(item["requested_rewrite"]["prompt"])
        else:
            raise ValueError(f"Unknown dataset name: {ds_name}. Use 'mcf' or 'zsre'.")
        
        # Remove duplicates while preserving order
        unique_subjects = list(dict.fromkeys(subjects))
        unique_prompts = list(dict.fromkeys(prompts))
        
        return unique_subjects, unique_prompts
        
    except Exception as e:
        print(f"Error loading dataset {ds_name}: {e}")
        import traceback
        traceback.print_exc()
        return [], []

