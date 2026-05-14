# [ICML'26 regular] Reverse-Engineering Model Editing on Language Models


> Reverse-Engineering Model Editing on Language Models,\
>Zhiyu Sun, Minrui Luo, Yu Wang, Zhili Chen, Tianxing He. *ICML, 2026*, [Link](https://arxiv.org/abs/2602.10134)
<p align="center">
  <img src="/resource/KSTER.png" width="90%">
</p>

## Requirements
**At least one A40 48G GPU.**

- torch==2.6.0
- einops==0.8.1
- higher==0.2.1
- hydra-core==1.3.2
- transformers==4.51.3
- datasets==2.21.0
- matplotlib==3.10.3
- spacy==3.4.1
- scipy==1.15.2
- scikit-learn==1.6.1
- nltk==3.9.1

## How to Run
Our main experiments live in `main_ablation_experiments/`.

### 1) Subject Inference Attack (Stage I)

Run (example):

```bash
python -m main_ablation_experiments.main_attack_experiment --experiment_type both

# Or run only batched editors (MEMIT/AlphaEdit)
python -m main_ablation_experiments.main_attack_experiment --experiment_type multi

# Or run only ROME (single-edit)
python -m main_ablation_experiments.main_attack_experiment --experiment_type single
```

### 2) Prompt Recovery Attack (Stage II)

Run (example):

```bash
python -m main_ablation_experiments.prompt_recovery_experiment
```

### 3) Subspace Camouflage Defense

Run (example):

```bash
python -m main_ablation_experiments.camouflage_scale_ablation --alg_name AlphaEdit_defence --model_name meta-llama/Meta-Llama-3-8B-Instruct --hparams_fname Llama3-8B.json --ds_name mcf --num_edits 100 --n_independent_runs 5 --scales 0,1,3,5
```

Furthermore, our important additional experiments during rebuttal live in `additional_experiments/`.

### 4) Candidate Subject Size Impact

Run (example):

```bash
# obtain IMDb dataset
python -m dsets.imbd_name
```
```bash
python -m additional_experiments.subjects_number_impact --model_name Llama3 --alg_name MEMIT --num_edits 100 --ds_name mcf --n_runs 5
```

### 5) Sequential Editing Scenario

Run (example):

```bash
python -m additional_experiments.sequential_subject_recovery_experiment --num_edits 100 --sequential_nums 10 --runs 3
```

### 6) Candidate Template Size Impact

Run (example):

```bash
python -m additional_experiments.prompt_number_impact --model_name Llama3 --alg_name MEMIT --num_edits 100 --ds_name mcf --n_runs 5
```

## Acknowledge

Our repo is built on AlphaEdit. We thank the authors for sharing their code.
