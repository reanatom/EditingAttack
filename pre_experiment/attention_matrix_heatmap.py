import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_subjects_and_templates():
    """
    创建10个主语和10个提示词模板
    """
    subjects = [
        "Albert Einstein",
        "Marie Curie",
        "Isaac Newton",
        "Galileo Galilei",
        "Charles Darwin",
        "Nikola Tesla",
        "Leonardo da Vinci",
        "Ada Lovelace",
        "Alan Turing",
        "Stephen Hawking"
    ]

    # templates = [
    #     "Considering their family background and early upbringing, the mother tongue of {} is",
    #     "According to historical records regarding their early life, the birthplace of {} is",
    #     "Throughout their long and distinguished career, the primary occupation of {} is",
    #     "Based on their legal citizenship and country of origin, the nationality of {} is",
    #     " regarding their personal spiritual beliefs and practices, the religion of {} is",
    #     "If we look at their specific area of professional expertise, the field of work of {} is",
    #     "At the end of their life journey, the recorded place of death of {} is",
    #     "During the peak of their professional career, the main employer of {} is",
    #     "Regarding their academic background and alma mater, the institution where {} was educated at is",
    #     "Among the many accolades and honors bestowed upon them, the award received by {} is"
    # ]

    templates = [
        "The mother tongue of {} is",
        "The birthplace of {} is",
        "The primary occupation of {} is",
        "The nationality of {} is",
        "The religion of {} is",
        "The field of work of {} is",
        "The recorded place of death of {} is",
        "The main employer of {} is",
        "The institution where {} was educated at is",
        "The award received by {} is"
    ]

    return subjects, templates


def get_attention_weights(model, tok, prompt, subject):
    """
    获取第4层注意力层的注意力权重，并找到主语的token位置
    
    Returns:
        attention_weights: shape (num_heads, seq_len, seq_len)
        subject_start_idx: 主语开始的token索引
        subject_end_idx: 主语结束的token索引
    """
    # 编码整个提示词
    full_prompt = prompt.format(subject)
    inputs = tok(full_prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    # 也单独编码主语部分来找到它的位置
    # 注意：Llama对空格敏感，在提示词中主语前有空格，所以这里也要加空格
    subject_tokens = tok(" " + subject, add_special_tokens=False)["input_ids"]

    # 在完整输入中找到主语的位置
    input_ids_list = input_ids[0].tolist()
    subject_len = len(subject_tokens)

    # 寻找主语token在完整序列中的位置
    subject_start_idx = None
    for i in range(len(input_ids_list) - subject_len + 1):
        if input_ids_list[i:i + subject_len] == subject_tokens:
            subject_start_idx = i
            break

    if subject_start_idx is None:
        # 如果没找到精确匹配，尝试找部分匹配
        # 有时tokenizer会因为前后空格产生不同的token
        # 我们用一个简单的方法：找第一个非特殊token作为开始
        subject_start_idx = 1  # 跳过BOS token
        subject_end_idx = subject_start_idx + subject_len - 1
    else:
        subject_end_idx = subject_start_idx + subject_len - 1

    # 获取模型输出，包括注意力权重
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            return_dict=True
        )

    # 获取第4层的注意力权重
    # outputs.attentions 是一个tuple，每层一个元素
    # 每个元素的shape是 (batch_size, num_heads, seq_len, seq_len)
    layer_4_attention = outputs.attentions[4]  # 第4层 (0-indexed)

    # 去掉batch维度
    attention_weights = layer_4_attention[0]  # shape: (num_heads, seq_len, seq_len)

    return attention_weights, subject_start_idx, subject_end_idx


def calculate_subject_attention_percentage(attention_weights, subject_start_idx, subject_end_idx):
    """
    计算主语last_token对主语部分的注意力权重百分比
    
    Args:
        attention_weights: shape (num_heads, seq_len, seq_len)
        subject_start_idx: 主语开始的token索引
        subject_end_idx: 主语结束的token索引（inclusive）
    
    Returns:
        percentage: 主语部分的注意力权重百分比
        bos_attention_percentage: BOS token的注意力权重百分比（用于诊断）
    """
    # 取主语最后一个token的注意力权重
    # attention_weights[:, subject_end_idx, :] 表示所有头、主语last_token对所有token的注意力
    # shape: (num_heads, seq_len)

    # 对所有头取平均
    avg_attention = attention_weights.mean(dim=0)  # shape: (seq_len, seq_len)

    # 取主语最后一个token对所有token的注意力
    last_token_attention = avg_attention[subject_end_idx, :]  # shape: (seq_len,)

    # 【关键修改】排除BOS token（索引0）的注意力权重
    # 计算对所有token的注意力总和（排除BOS）
    total_attention_excluding_bos = last_token_attention[1:].sum().item()

    # 保存BOS token的注意力权重（用于诊断）
    bos_attention = last_token_attention[0].item()

    # 计算对主语部分token的注意力总和
    subject_attention = last_token_attention[subject_start_idx:subject_end_idx + 1].sum().item()

    # 计算百分比（排除BOS后）
    if total_attention_excluding_bos > 0:
        percentage = (subject_attention / total_attention_excluding_bos) * 100
    else:
        percentage = 0.0

    return percentage, bos_attention


def main():
    print("=" * 80)
    print("Attention Matrix Heatmap Generation")
    print("=" * 80)

    # 1. 创建主语和模板
    subjects, templates = create_subjects_and_templates()
    print(f"Created {len(subjects)} subjects and {len(templates)} templates")
    print(f"Total samples: {len(subjects) * len(templates)}")

    # 2. 加载模型
    print("\nLoading model and tokenizer...")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 3. 创建结果矩阵
    # 行: 模板索引 (0-9)
    # 列: 主语索引 (0-9)
    percentage_matrix = np.zeros((len(templates), len(subjects)))
    bos_attention_matrix = np.zeros((len(templates), len(subjects)))  # 用于诊断BOS注意力

    # 4. 对每个组合进行处理
    print("\nProcessing samples...")
    total_samples = len(subjects) * len(templates)
    sample_count = 0

    for template_idx, template in enumerate(templates):
        for subject_idx, subject in enumerate(subjects):
            sample_count += 1
            print(f"\rProcessing sample {sample_count}/{total_samples} "
                  f"(Template {template_idx}, Subject {subject_idx})", end="", flush=True)

            try:
                # 获取注意力权重和主语位置
                attention_weights, subject_start, subject_end = get_attention_weights(
                    model, tok, template, subject
                )

                # 计算主语部分的注意力百分比（排除BOS）
                percentage, bos_attn = calculate_subject_attention_percentage(
                    attention_weights, subject_start, subject_end
                )

                # 存储结果
                percentage_matrix[template_idx, subject_idx] = percentage
                bos_attention_matrix[template_idx, subject_idx] = bos_attn

            except Exception as e:
                print(f"\nError processing Template {template_idx}, Subject {subject_idx}: {e}")
                percentage_matrix[template_idx, subject_idx] = 0.0
                bos_attention_matrix[template_idx, subject_idx] = 0.0

    print("\nProcessing completed!")

    # 5. 绘制热力图（符合论文规范）
    print("\nGenerating heatmap...")

    # 设置字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 设置图形大小：3.25英寸 x 3.25英寸（单栏论文）
    fig, ax = plt.subplots(figsize=(3.25, 3.25))

    # 主语注意力百分比热力图（排除BOS）
    sns.heatmap(
        percentage_matrix,
        annot=True,  # 显示数值
        fmt='.1f',  # 数值格式：保留1位小数（节省空间）
        cmap='YlOrRd',  # 黄-橙-红色图
        cbar_kws={'label': 'Attention (%)'},
        xticklabels=[f'S{i}' for i in range(len(subjects))],  # 主语索引
        yticklabels=[f'T{i}' for i in range(len(templates))],  # 模板索引
        linewidths=0.3,  # 网格线宽度（更细）
        linecolor='white',  # 网格线颜色
        square=True,
        ax=ax,
        annot_kws={'fontsize': 6, 'family': 'Times New Roman'}  # 格子内数字：6pt
    )

    # 设置坐标轴标签（10pt）
    ax.set_xlabel('Subject Index', fontsize=10, family='Times New Roman')
    ax.set_ylabel('Template Index', fontsize=10, family='Times New Roman')

    # 设置刻度标签（8pt）
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 设置colorbar标签字体
    cbar = ax.collections[0].colorbar
    cbar.set_label('Attention (%)', fontsize=10, family='Times New Roman')
    cbar.ax.tick_params(labelsize=8)

    # 调整布局，确保所有元素都在图内
    plt.tight_layout()

    # 6. 保存为PDF，300 DPI
    output_path = "pre_experiment/attention_heatmap_short.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")

    # 7. 打印统计信息
    print("\n" + "=" * 80)
    print("Statistics (Subject Attention, BOS Excluded):")
    print(f"Mean attention percentage: {percentage_matrix.mean():.2f}%")
    print(f"Std attention percentage: {percentage_matrix.std():.2f}%")
    print(f"Min attention percentage: {percentage_matrix.min():.2f}%")
    print(f"Max attention percentage: {percentage_matrix.max():.2f}%")
    print("\nStatistics (BOS Attention Sink):")
    print(f"Mean BOS attention weight: {bos_attention_matrix.mean():.4f}")
    print(f"Std BOS attention weight: {bos_attention_matrix.std():.4f}")
    print(f"Min BOS attention weight: {bos_attention_matrix.min():.4f}")
    print(f"Max BOS attention weight: {bos_attention_matrix.max():.4f}")
    print("=" * 80)

    # 8. 保存原始数据
    np.save("pre_experiment/attention_percentage_matrix.npy", percentage_matrix)
    np.save("pre_experiment/bos_attention_matrix.npy", bos_attention_matrix)
    print(f"Raw data saved to: pre_experiment/attention_percentage_matrix.npy")
    print(f"BOS attention data saved to: pre_experiment/bos_attention_matrix.npy")

    # 9. 保存主语和模板信息
    with open("pre_experiment/subjects_and_templates.txt", "w", encoding="utf-8") as f:
        f.write("Subjects:\n")
        for i, subj in enumerate(subjects):
            f.write(f"  S{i}: {subj}\n")
        f.write("\nTemplates:\n")
        for i, tmpl in enumerate(templates):
            f.write(f"  T{i}: {tmpl}\n")
    print(f"Subjects and templates saved to: pre_experiment/subjects_and_templates.txt")

    print("\nAll done!")


if __name__ == "__main__":
    main()
