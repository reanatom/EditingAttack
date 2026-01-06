# 协方差消融实验修正说明

## 🐛 发现的问题

原代码使用了 `load_dataset_data()` 来加载数据，但这个函数返回的数据格式与 `apply_memit_to_model` 期望的格式不一致。

### 错误的实现

```python
# ❌ 错误：使用 load_dataset_data
from util.data_loader import load_dataset_data

all_subjects, all_data = load_dataset_data(limit=2000)
indices = np.random.choice(len(all_data), size=num_edits, replace=False)
edit_data = [all_data[i] for i in indices]

# 传给 apply_memit_to_model
edited_model, _ = apply_memit_to_model(
    model=model,
    tok=tok,
    requests=edit_data,  # 格式不对！
    hparams=base_hparams,
)
```

**问题**：
- `load_dataset_data` 返回的是简单的字典列表，缺少必要的结构
- 缺少 `requested_rewrite` 嵌套结构
- 可能缺少必要的字段如 `prompt`, `target_new` 等

### 正确的实现

```python
# ✅ 正确：使用 MultiCounterFactDataset（与 camouflage_scale_ablation.py 一致）
from dsets import MultiCounterFactDataset

ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
ds_list = list(ds)

# 随机采样
import random
random.seed(run_id * 12345)
sampled_records = random.sample(ds_list, num_edits)

# 提取真实主语
true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]

# 准备 request 格式
edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]

# 传给 apply_memit_to_model
edited_model, _ = apply_memit_to_model(
    model=model,
    tok=tok,
    requests=edit_data,  # 格式正确！
    hparams=base_hparams,
)
```

## 📋 数据格式详解

### MultiCounterFactDataset 返回的格式

```python
{
    "case_id": "12345",
    "requested_rewrite": {
        "subject": "Albert Einstein",
        "prompt": "The mother tongue of {} is",
        "target_new": {
            "str": " German"
        },
        "target_true": {
            "str": " Jewish"
        },
        "relation_id": "P103",
        # ... 其他字段
    }
}
```

### apply_memit_to_model 期望的格式

```python
[
    {
        "case_id": "12345",
        "subject": "Albert Einstein",
        "prompt": "The mother tongue of {} is",
        "target_new": {
            "str": " German"
        },
        # ... 其他字段（来自 requested_rewrite）
    },
    # ... 更多样本
]
```

通过 `{"case_id": r["case_id"], **r["requested_rewrite"]}` 可以正确展开。

## ✅ 修改内容总结

### 1. 导入修改

```python
# 修改前
from util.data_loader import load_dataset_data

# 修改后
from dsets import MultiCounterFactDataset
```

### 2. 数据加载修改

```python
# 修改前
all_subjects, all_data = load_dataset_data(limit=2000)
indices = np.random.choice(len(all_data), size=num_edits, replace=False)
edit_data = [all_data[i] for i in indices]
true_subjects = [all_data[i]["subject"] for i in indices]

# 修改后
ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=2000)
ds_list = list(ds)
random.seed(run_id * 12345)
sampled_records = random.sample(ds_list, num_edits)
true_subjects = [r["requested_rewrite"]["subject"] for r in sampled_records]
edit_data = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in sampled_records]
# case_id 保持不变
```

### 3. create_name_database 保持不变

`create_name_database()` 函数仍然使用 `load_dataset_data`，因为它只需要候选主语列表，不需要完整的数据结构：

```python
def create_name_database():
    from util.data_loader import load_dataset_data
    subjects, _ = load_dataset_data(limit=2000)
    return subjects
```

## 🔍 为什么这个区别很重要？

1. **字段完整性**: `MultiCounterFactDataset` 确保所有必需字段都存在
2. **格式一致性**: 与 `camouflage_scale_ablation.py` 保持一致，便于对比
3. **可重现性**: 使用标准的数据集类，确保实验可重现
4. **调试便利**: 如果出现问题，更容易追踪数据流

## 🎯 验证修正

运行实验前，可以通过以下方式验证数据格式：

```python
from dsets import MultiCounterFactDataset
from util.globals import DATA_DIR

# 加载数据
ds = MultiCounterFactDataset(DATA_DIR, tok=tok, size=10)
sample = ds[0]

# 检查格式
print("Keys:", sample.keys())
print("Requested rewrite keys:", sample["requested_rewrite"].keys())
print("Subject:", sample["requested_rewrite"]["subject"])
print("Prompt:", sample["requested_rewrite"]["prompt"])
print("Target:", sample["requested_rewrite"]["target_new"]["str"])
```

期望输出：
```
Keys: dict_keys(['case_id', 'requested_rewrite', ...])
Requested rewrite keys: dict_keys(['subject', 'prompt', 'target_new', 'target_true', ...])
Subject: Albert Einstein
Prompt: The mother tongue of {} is
Target:  German
```

## 📚 相关文件

- `main_ablation_experiments/covariance_ablation.py` - 主实验脚本（已修正）
- `main_ablation_experiments/camouflage_scale_ablation.py` - 参考实现
- `dsets/__init__.py` - MultiCounterFactDataset 定义
- `util/data_loader.py` - load_dataset_data 定义（用于候选主语）

## ⚠️ 注意事项

修正后，确保：
1. 检查 `DATA_DIR` 路径是否正确
2. 确保 `MultiCounterFactDataset` 数据集已经准备好
3. 注意不同实验可能会使用相同的 `case_id`，编辑量文件可能会被覆盖（这是正常的）

---

**修正完成日期**: 2025-12-23
**修正原因**: 用户指出数据加载方式与 camouflage_scale_ablation.py 不一致
**验证状态**: 代码通过 linter 检查，等待实际运行验证

