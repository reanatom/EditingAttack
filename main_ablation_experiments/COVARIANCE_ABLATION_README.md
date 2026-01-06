# 协方差矩阵估计消融实验

## 📋 实验目标

测试使用不同数据集和不同采样大小计算的协方差矩阵对主语攻击效果的影响。

## ⚙️ 实验设置

### 固定参数
- **模型**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **算法**: MEMIT
- **编辑数量**: 100个样本/批次
- **独立实验次数**: 5次
- **目标层**: Layer 4

### 变量参数（协方差矩阵估计）
- **数据集**: `wikipedia`, `wikitext`, `pile`
- **采样大小**: `10`, `100`, `1000`, `10000`

### 评估指标
1. **Top-100召回率**: 真实主语在前100名候选中的比例
2. **平均投影分数**: 真实主语在恢复子空间上的投影分数

## 🚀 使用方法

### 1. 运行完整实验

```bash
cd E:\My_Project\AlphaEdit
python main_ablation_experiments/covariance_ablation.py
```

**注意**: 
- 总计将运行 **3 × 4 × 5 = 60** 次独立实验
- 每次实验包括: 编辑100个样本 + 主语攻击
- 预计总时长: **10-20小时**（取决于GPU性能）

### 2. 分析结果

实验完成后，运行分析脚本：

```bash
python main_ablation_experiments/analyze_covariance_results.py
```

这将生成：
- 详细统计信息（终端输出）
- 热力图可视化（PDF）
- 折线图可视化（PDF）

## 📊 输出文件

### 主要结果
```
main_ablation_experiments/covariance_results/
├── covariance_ablation_results.xlsx  # Excel表格（两个sheet）
│   ├── Top100_Recall                 # 召回率
│   └── Avg_Projection_Score          # 投影分数
├── recall_results.csv                # 召回率（CSV备份）
├── proj_score_results.csv            # 投影分数（CSV备份）
└── intermediate_results.json         # 中间结果（所有原始数据）
```

### 可视化结果
```
main_ablation_experiments/covariance_results/
├── covariance_heatmaps.pdf           # 热力图（2个子图）
└── covariance_lineplots.pdf          # 折线图（2个子图）
```

## 📈 结果格式

### Excel表格示例

**Sheet 1: Top100_Recall**

| Dataset    | Size=10        | Size=100       | Size=1000      | Size=10000     |
|------------|----------------|----------------|----------------|----------------|
| wikipedia  | 0.8520 ± 0.012 | 0.8940 ± 0.008 | 0.9120 ± 0.005 | 0.9350 ± 0.004 |
| wikitext   | 0.8310 ± 0.015 | 0.8850 ± 0.010 | 0.9080 ± 0.006 | 0.9320 ± 0.005 |
| pile       | 0.8480 ± 0.013 | 0.8900 ± 0.009 | 0.9100 ± 0.005 | 0.9340 ± 0.004 |

**Sheet 2: Avg_Projection_Score**

| Dataset    | Size=10          | Size=100         | Size=1000        | Size=10000       |
|------------|------------------|------------------|------------------|------------------|
| wikipedia  | 0.756123 ± 0.012 | 0.812345 ± 0.008 | 0.845678 ± 0.005 | 0.867890 ± 0.004 |
| wikitext   | 0.745123 ± 0.015 | 0.805678 ± 0.010 | 0.838901 ± 0.006 | 0.861234 ± 0.005 |
| pile       | 0.750456 ± 0.013 | 0.809012 ± 0.009 | 0.842345 ± 0.005 | 0.865678 ± 0.004 |

## 🔧 实验流程详解

### 单次独立实验流程

对于每个 `(数据集, 采样大小)` 组合，进行5次独立实验：

```python
for run_id in range(5):
    # 1. 加载模型
    model = load_model()
    
    # 2. 从 MultiCounterFactDataset 随机采样100个样本（使用不同的随机种子）
    ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
    ds_list = list(ds)
    sampled_records = random.sample(ds_list, num=100)
    edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]
    
    # 3. 执行MEMIT编辑
    edited_model = apply_memit(model, edit_data)
    
    # 4. 配置攻击参数（使用指定的协方差估计设置）
    config.mom2_dataset = cov_dataset      # 例如 "wikipedia"
    config.mom2_n_samples = cov_sample_size  # 例如 1000
    
    # 5. 执行主语攻击
    Q_basis = recover_subspace(edited_model, config)
    
    # 6. 计算所有候选主语的投影分数并排序
    scores = rank_candidates(Q_basis, candidate_database)
    
    # 7. 计算指标
    recall = count_true_subjects_in_top100(scores) / 100
    proj_score = average_score_of_true_subjects(scores)
    
    # 8. 记录结果
    results[dataset][size].append((recall, proj_score))
```

## 📌 注意事项

### 1. 数据格式

实验使用 `MultiCounterFactDataset` 加载数据，与 `camouflage_scale_ablation.py` 保持一致：

```python
ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
ds_list = list(ds)
sampled_records = random.sample(ds_list, num_edits)

# 每个 record 的格式：
# {
#   "case_id": "...",
#   "requested_rewrite": {
#     "subject": "Albert Einstein",
#     "prompt": "The mother tongue of {} is",
#     "target_new": {"str": " German"},
#     ...
#   }
# }

# 传给 apply_memit_to_model 的格式：
edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]
```

### 2. 协方差矩阵计算位置

协方差矩阵在 `attack_kr_opt.py` 的 `get_cov` 函数中计算，其参数来自攻击配置：

```python
C_matrix = get_cov(
    model, tok, layer_name,
    mom2_dataset=config.mom2_dataset,      # 由实验脚本指定
    mom2_n_samples=config.mom2_n_samples,  # 由实验脚本指定
    mom2_dtype=config.mom2_dtype
)
```

### 3. 缓存机制

- 协方差矩阵会被缓存在 `experiments/attack_kr_opt.py` 的 `COV_CACHE` 中
- **重要**: 每次切换数据集或采样大小时，需要重新加载模型以清空缓存

### 4. 内存管理

每次独立实验后会清理GPU内存：

```python
del model
del edited_model
torch.cuda.empty_cache()
```

### 5. 中间结果保存

每完成一个 `(数据集, 采样大小)` 组合的5次实验后，会立即保存中间结果，防止实验中断导致数据丢失。

### 6. 知识模板固定

为了实验的一致性，所有攻击使用统一的知识模板：
```python
knowledge_template = "The mother tongue of {} is"
```

**注意**: 这可能与某些样本的实际关系类型不匹配，但由于我们关注的是协方差估计对攻击效果的影响（相对比较），这不会影响实验结论。

### 7. case_id 处理

实验直接使用 `MultiCounterFactDataset` 中原始的 `case_id`，不做修改。不同实验轮次可能会使用相同的 `case_id`，这会导致编辑量文件被覆盖，但这不影响实验结果，因为：
- 每次实验都是独立的（重新加载模型）
- 攻击立即在编辑后执行，使用当前的编辑量文件
- 我们关注的是最终的统计结果，不需要保留所有中间文件

## 🐛 故障排查

### 问题1: CUDA OOM（显存不足）

**解决方案**:
- 减少 `num_edits`（从100改为50）
- 或者在配置中设置更小的 `target_rank`

### 问题2: 某个数据集无法加载

**可能原因**:
- `wikitext` 或 `pile` 数据集尚未下载
- 需要先运行 `rome/layer_stats.py` 确保数据集可用

**解决方案**:
```bash
# 测试数据集是否可用
python -c "from util.data_loader import load_dataset_data; print('OK')"
```

### 问题3: 实验中断后如何继续

实验会自动保存中间结果到 `intermediate_results.json`。如需继续：

1. 修改代码跳过已完成的实验
2. 或者直接从中间结果文件生成表格：

```bash
python main_ablation_experiments/analyze_covariance_results.py
```

## 📚 相关文件

- `experiments/attack_kr_opt.py`: 包含 `get_cov` 函数和协方差矩阵加载逻辑
- `experiments/attack_memit_recovery_rank_n.py`: 主语攻击实现（秩>1）
- `memit/memit_main.py`: MEMIT编辑实现
- `util/data_loader.py`: 数据集加载

## 🎯 预期发现

基于实验假设，我们预期：

1. **采样大小影响**: 
   - 采样大小越大 → 协方差估计越准确 → 攻击效果越好
   - Top-100召回率和投影分数应随采样大小增加而提高

2. **数据集影响**:
   - `wikipedia`: 最接近编辑数据分布，预期效果最好
   - `wikitext` 和 `pile`: 效果可能略差，但差异不应太大

3. **收敛趋势**:
   - 采样大小 > 1000 后，性能提升应趋于平缓（收益递减）

## 📞 联系方式

如有问题，请检查：
- 终端输出日志
- `intermediate_results.json` 文件
- GPU内存使用情况（`nvidia-smi`）

