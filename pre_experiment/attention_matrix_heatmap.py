import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_subjects_and_templates():

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


    full_prompt = prompt.format(subject)
    inputs = tok(full_prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]


    subject_tokens = tok(" " + subject, add_special_tokens=False)["input_ids"]


    input_ids_list = input_ids[0].tolist()
    subject_len = len(subject_tokens)


    subject_start_idx = None
    for i in range(len(input_ids_list) - subject_len + 1):
        if input_ids_list[i:i + subject_len] == subject_tokens:
            subject_start_idx = i
            break

    if subject_start_idx is None:

        subject_start_idx = 1
        subject_end_idx = subject_start_idx + subject_len - 1
    else:
        subject_end_idx = subject_start_idx + subject_len - 1


    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            return_dict=True
        )


    layer_4_attention = outputs.attentions[4]


    attention_weights = layer_4_attention[0]  # shape: (num_heads, seq_len, seq_len)

    return attention_weights, subject_start_idx, subject_end_idx


def calculate_subject_attention_percentage(attention_weights, subject_start_idx, subject_end_idx):



    avg_attention = attention_weights.mean(dim=0)  # shape: (seq_len, seq_len)


    last_token_attention = avg_attention[subject_end_idx, :]  # shape: (seq_len,)


    total_attention_excluding_bos = last_token_attention[1:].sum().item()


    bos_attention = last_token_attention[0].item()


    subject_attention = last_token_attention[subject_start_idx:subject_end_idx + 1].sum().item()


    if total_attention_excluding_bos > 0:
        percentage = (subject_attention / total_attention_excluding_bos) * 100
    else:
        percentage = 0.0

    return percentage, bos_attention


def main():
    print("=" * 80)
    print("Attention Matrix Heatmap Generation")
    print("=" * 80)


    subjects, templates = create_subjects_and_templates()
    print(f"Created {len(subjects)} subjects and {len(templates)} templates")
    print(f"Total samples: {len(subjects) * len(templates)}")


    print("\nLoading model and tokenizer...")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token


    percentage_matrix = np.zeros((len(templates), len(subjects)))
    bos_attention_matrix = np.zeros((len(templates), len(subjects)))


    print("\nProcessing samples...")
    total_samples = len(subjects) * len(templates)
    sample_count = 0

    for template_idx, template in enumerate(templates):
        for subject_idx, subject in enumerate(subjects):
            sample_count += 1
            print(f"\rProcessing sample {sample_count}/{total_samples} "
                  f"(Template {template_idx}, Subject {subject_idx})", end="", flush=True)

            try:

                attention_weights, subject_start, subject_end = get_attention_weights(
                    model, tok, template, subject
                )


                percentage, bos_attn = calculate_subject_attention_percentage(
                    attention_weights, subject_start, subject_end
                )


                percentage_matrix[template_idx, subject_idx] = percentage
                bos_attention_matrix[template_idx, subject_idx] = bos_attn

            except Exception as e:
                print(f"\nError processing Template {template_idx}, Subject {subject_idx}: {e}")
                percentage_matrix[template_idx, subject_idx] = 0.0
                bos_attention_matrix[template_idx, subject_idx] = 0.0

    print("\nProcessing completed!")


    print("\nGenerating heatmap...")


    plt.rcParams['font.family'] = 'Times New Roman'
    

    fig, ax = plt.subplots(figsize=(3.25, 3.25))


    sns.heatmap(
        percentage_matrix,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention (%)'},
        xticklabels=[f'S{i}' for i in range(len(subjects))],
        yticklabels=[f'T{i}' for i in range(len(templates))],
        linewidths=0.3,
        linecolor='white',
        square=True,
        ax=ax,
        annot_kws={'fontsize': 6, 'family': 'Times New Roman'}
    )


    ax.set_xlabel('Subject Index', fontsize=10, family='Times New Roman')
    ax.set_ylabel('Template Index', fontsize=10, family='Times New Roman')


    ax.tick_params(axis='both', which='major', labelsize=8)
    

    cbar = ax.collections[0].colorbar
    cbar.set_label('Attention (%)', fontsize=10, family='Times New Roman')
    cbar.ax.tick_params(labelsize=8)


    plt.tight_layout()


    output_path = "pre_experiment/attention_heatmap_short.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")


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


    np.save("pre_experiment/attention_percentage_matrix.npy", percentage_matrix)
    np.save("pre_experiment/bos_attention_matrix.npy", bos_attention_matrix)
    print(f"Raw data saved to: pre_experiment/attention_percentage_matrix.npy")
    print(f"BOS attention data saved to: pre_experiment/bos_attention_matrix.npy")


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
