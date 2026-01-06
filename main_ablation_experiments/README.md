# Main Ablation Experiments

This directory contains code for running ablation studies on the defence algorithms.

## Camouflage Scale Ablation

### Overview
`camouflage_scale_ablation.py` evaluates the trade-off between model utility and privacy as the `camouflage_scale` parameter increases from 1 to 10.

### Key Features

1. **Dynamic Parameter Setting**: The script dynamically sets `camouflage_scale` in hparams, which is correctly read by the defence algorithms (`memit_orth_main.py`, `rome_orth_main.py`, `AlphaEdit_orth_main.py`).

2. **Proper P Matrix Computation**: For AlphaEdit, the projection matrix P is computed using `get_project()` (null space projection from covariance matrix), not a naive identity matrix.

3. **Independent Runs**: Each scale is tested with multiple independent editing runs to ensure statistical significance.

4. **Comprehensive Evaluation**: 
   - **Model Utility**: GLUE benchmarks (NLI, SST, CoLA, RTE, MMLU, MRPC)
   - **Privacy**: Attack-based evaluation using appropriate attack scripts for each algorithm

### Usage

```bash
python main_ablation_experiments/camouflage_scale_ablation.py \
    --alg_name MEMIT_defence \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname Llama3-8B.json \
    --dataset_size_limit 2000 \
    --num_edits 10 \
    --n_independent_runs 5 \
    --scales 1,2,3,4,5,6,7,8,9,10
```

### Parameters

- `--alg_name`: Defence algorithm (ROME_defence, MEMIT_defence, AlphaEdit_defence)
- `--model_name`: Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)
- `--hparams_fname`: Hyperparameter file (default: Llama3-8B.json)
- `--dataset_size_limit`: Maximum dataset size (default: 2000)
- `--num_edits`: Number of edits per run (default: 10, ROME always uses 1)
- `--n_independent_runs`: Number of independent runs per scale (default: 5)
- `--scales`: Comma-separated camouflage scales to test (default: 1-10)

### Output Structure

Results are saved to `main_ablation_experiments/results/`:

```
main_ablation_experiments/
└── results/
    ├── camouflage_scale=1/
    │   ├── MEMIT_defence/
    │   │   ├── edit0/
    │   │   │   └── glue_results.json
    │   │   ├── edit1/
    │   │   │   └── glue_results.json
    │   │   └── privacy_rank.json
    │   └── ...
    ├── camouflage_scale=2/
    │   └── ...
    └── ...
```

- **`edit{i}/glue_results.json`**: GLUE evaluation results for run i
  - Contains scores for: NLI, SST-2, CoLA, RTE, MMLU, MRPC
  - Each with accuracy/F1 scores
  
- **`privacy_rank.json`**: Privacy attack results
  - `camouflage_scale`: The scale value
  - `individual_run_avg_ranks`: Average rank for each independent run
  - `overall_average_rank`: Mean of all runs for this scale
  - `n_runs`: Number of successful runs

### Example Runs

**MEMIT Defence:**
```bash
python main_ablation_experiments/camouflage_scale_ablation.py \
    --alg_name MEMIT_defence \
    --num_edits 10 \
    --n_independent_runs 3 \
    --scales 1,5,10
```

**ROME Defence:**
```bash
python main_ablation_experiments/camouflage_scale_ablation.py \
    --alg_name ROME_defence \
    --n_independent_runs 5 \
    --scales 1,2,3,4,5
```

**AlphaEdit Defence:**
```bash
python main_ablation_experiments/camouflage_scale_ablation.py \
    --alg_name AlphaEdit_defence \
    --num_edits 10 \
    --n_independent_runs 3 \
    --scales 5,6,7,8,9,10
```

### Implementation Details

#### 1. Camouflage Scale Integration

The defence algorithms have been modified to read `camouflage_scale` from hparams:

```python
# In memit_orth_main.py, rome_orth_main.py, AlphaEdit_orth_main.py
camouflage_scale = getattr(hparams, 'camouflage_scale', 5)  # Default: 5
```

This ensures:
- If hparams has `camouflage_scale`, it's used
- Otherwise, default value (5) is used
- No errors even if the JSON doesn't define this parameter

#### 2. AlphaEdit P Matrix (Pre-computed Once)

**IMPORTANT**: The P matrix is computed **only once** before the main experiment loop, not for each scale or each run.

```python
# Pre-compute P matrices (before camouflage_scales loop)
if "AlphaEdit" in alg_name:
    print("Pre-computing P matrices for AlphaEdit...")
    for i, layer in enumerate(base_hparams.layers):
        P[i,:,:] = get_project(model, tok, layer, base_hparams)
    # P is then reused for ALL scales and ALL runs
```

Why this matters:
- Computing covariance matrix requires ~100K samples (several minutes)
- P depends only on the base model, NOT on edits or scales
- Pre-computing saves hours of redundant computation
- For 10 scales × 5 runs = 50 runs, this saves 49× computation time

The P matrix uses null space projection from the covariance matrix:

```python
def get_project(model, tok, layer, hparams):
    cov = get_cov(...)  # Get covariance matrix (slow!)
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
```

#### 3. Privacy Evaluation (True Subject Ranking)

Each algorithm uses its appropriate attack script with **true subject tracking**:

```python
# Extract true subjects from sampled records
true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]

# Pass true subjects to attack functions
ranks = attack_memit(..., true_subjects=[subject])
# or
ranks = attack_alphaedit(..., true_subjects=[subject])
# or  
ranks = attack_rome(..., true_subjects=[subject])
```

The attack functions:
- Score ALL candidate subjects from the database
- Find the **actual rank** of the true edited subject(s)
- Return real ranks (e.g., [1], [523], [1847]) instead of placeholder values

This gives meaningful privacy metrics:
- **Lower rank** = attack succeeds (bad for privacy)
- **Higher rank** = subject hidden among many candidates (good for privacy)
- **Average rank across runs** = overall privacy measure for that scale

### Notes

1. **ROME_defence** always edits one sample at a time, regardless of `--num_edits`
2. The script requires pre-computed statistics (covariance matrices, etc.) to be available
3. Each independent run uses:
   - A fresh model instance
   - Random sample from the dataset
   - Same `camouflage_scale` value
4. Results are cumulative across all independent runs for statistical robustness

### Troubleshooting

**Q: Getting "camouflage_scale not found in hparams" error?**  
A: This shouldn't happen with `getattr(hparams, 'camouflage_scale', default)`. Check that you're using the updated defence algorithms.

**Q: AlphaEdit results are poor?**  
A: Ensure P matrices are being computed correctly with `get_project()`, not using identity matrices.

**Q: Out of memory errors?**  
A: Reduce `--n_independent_runs` or `--num_edits`, or use a smaller model/dataset.

**Q: Attack evaluation returns empty ranks?**  
A: Check that the edit amounts files are being saved correctly by the defence algorithms.

## Results Aggregation

### Overview
`aggregate_results.py` aggregates experimental results into two Excel tables:

1. **Privacy Table** (`privacy_runs={n}.xlsx`): Average rank of true subjects for each algorithm and scale
2. **Effect Table** (`effect_runs={n}.xlsx`): GLUE F1 scores (mean ± std) for each dataset, algorithm, and scale

### Usage

After running the ablation experiments, aggregate results:

```bash
python main_ablation_experiments/aggregate_results.py
```

This will automatically:
1. Scan `main_ablation_experiments/results/` directory
2. Collect data from all `camouflage_scale=X` directories
3. Generate two Excel files with formatted results

### Output Tables

#### Privacy Table (`privacy_runs={n}.xlsx`)

| camouflage_scale | MEMIT_defence | ROME_defence | AlphaEdit_defence |
|------------------|---------------|--------------|-------------------|
| 1                | 15.46         | 342.21       | 523.11            |
| 2                | 23.89         | 456.78       | 678.45            |
| ...              | ...           | ...          | ...               |

- **Rows**: Different camouflage scale values
- **Columns**: Defence algorithms
- **Values**: Overall average rank from `privacy_rank.json`
- **Higher rank** = Better privacy (true subject harder to find)

#### Effect Table (`effect_runs={n}.xlsx`)

Multi-level header structure:

| | **sst** | | | **mmmlu** | | | ... |
|---|---------|---|---|-----------|---|---|-----|
| **camouflage_scale** | **MEMIT** | **ROME** | **AlphaEdit** | **MEMIT** | **ROME** | **AlphaEdit** | ... |
| 1 | 0.8311±0.0123 | 0.8456±0.0098 | 0.8234±0.0156 | ... | ... | ... | ... |
| 2 | 0.8289±0.0145 | 0.8423±0.0112 | 0.8198±0.0167 | ... | ... | ... | ... |

- **Rows**: Different camouflage scale values
- **Top-level columns**: GLUE datasets (sst, mmmlu, mrpc, cola, rte, nli)
- **Sub-columns**: Defence algorithms
- **Values**: F1 score formatted as `mean ± std`
  - Mean: Average F1 across all independent runs
  - Std: Standard deviation across runs

### Data Sources

The script reads:

1. **Privacy data**: `{scale}/{algorithm}/privacy_rank.json`
   ```json
   {
       "camouflage_scale": 1,
       "individual_run_avg_ranks": [15.5, 15.5, 15.4, 15.4, 15.5],
       "overall_average_rank": 15.46,
       "n_runs": 5
   }
   ```

2. **Effect data**: `{scale}/{algorithm}/edit{i}/glue_results.json`
   ```json
   {
       "edit_num": 0,
       "camouflage_scale": 1,
       "sst": {"f1": 0.8311, ...},
       "mmmlu": {"f1": 0.5541, ...},
       ...
   }
   ```

### Implementation Details

#### Key Functions

1. **`extract_scale_from_dirname(dirname)`**
   - Parses directory name to extract scale value
   - Example: `'camouflage_scale=5'` → `5`

2. **`load_privacy_data(alg_dir)`**
   - Loads `privacy_rank.json`
   - Returns `overall_average_rank`

3. **`load_glue_scores(alg_dir, datasets)`**
   - Scans all `edit*` directories
   - Extracts F1 scores for each dataset
   - Computes mean and standard deviation
   - Returns: `{dataset: (mean, std)}`

4. **`aggregate_results()`**
   - Main orchestration function
   - Creates both tables
   - Saves to Excel with proper formatting

#### Multi-level Headers

The Effect table uses pandas MultiIndex for hierarchical columns:

```python
multi_columns = [
    ('sst', 'MEMIT_defence'),
    ('sst', 'ROME_defence'),
    ('sst', 'AlphaEdit_defence'),
    ('mmmlu', 'MEMIT_defence'),
    ...
]
multi_index = pd.MultiIndex.from_tuples(multi_columns, names=["Dataset", "Algorithm"])
```

This creates Excel columns grouped by dataset, with sub-columns for each algorithm.

### Requirements

Make sure you have the required Python packages:

```bash
pip install pandas openpyxl numpy
```

### Example Output

After running aggregation:

```
================================================================================
Aggregating Camouflage Scale Ablation Results
================================================================================
Found scales: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Found algorithms: ['AlphaEdit_defence', 'MEMIT_defence', 'ROME_defence']
Number of independent runs per scale: 5

Generating Privacy Table...
Privacy table saved to: main_ablation_experiments/privacy_runs=5.xlsx
                  AlphaEdit_defence  MEMIT_defence  ROME_defence
camouflage_scale                                                
1                           523.112         15.460       342.210
2                           678.453         23.890       456.780
...

Generating Effect Table...
Effect table saved to: main_ablation_experiments/effect_runs=5.xlsx
...

================================================================================
Aggregation Complete!
================================================================================
Privacy Table: main_ablation_experiments/privacy_runs=5.xlsx
Effect Table: main_ablation_experiments/effect_runs=5.xlsx
```

### Citation

If you use this ablation study in your research, please cite the original AlphaEdit paper and mention the camouflage scale defence mechanism.
