import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import softmax
import os
from transformers import (
    BertTokenizer, BertForMaskedLM,
    DistilBertTokenizer, DistilBertForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM,
    T5Tokenizer, T5ForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModel
)

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").eval()

distil_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distil_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased").eval()

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForMaskedLM.from_pretrained("roberta-base").eval()

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").eval()

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
# ---------------------------------------------------------
# Utility: extract bucket → matrix for a single metric
# ---------------------------------------------------------
import numpy as np
from collections import defaultdict

def build_metric_matrix(all_results, metric):
    """
    Returns:
        analogy_types: list of row names
        models: list of column names
        mask_positions: ['beginning', 'middle', 'end']
        matrices: dict of mask_position -> 2D numpy array (averaged over examples)
    """
    analogy_types = sorted(list({x["analogy_type"] for x in all_results}))
    models = ["BERT", "DistilBERT", "RoBERTa", "T5", "GPT-2"]
    mask_positions = ["beginning", "middle", "end"]

    # Use a nested structure to collect all metric values
    temp_values = {mp: {r: {m: [] for m in models} for r in analogy_types} for mp in mask_positions}

    # Collect values from all results
    for r in all_results:
        row = r["analogy_type"]
        mp = r["mask_position"]
        for m in models:
            if m in r:
                temp_values[mp][row][m].append(r[m][metric])

    # Compute averages and build matrices
    matrices = {}
    for mp in mask_positions:
        mat = np.zeros((len(analogy_types), len(models)))
        for i, row in enumerate(analogy_types):
            for j, m in enumerate(models):
                vals = temp_values[mp][row][m]
                mat[i, j] = np.mean(vals) if vals else 0.0
        matrices[mp] = mat

    return analogy_types, models, mask_positions, matrices


# ---------------------------------------------------------
# Layout 1: ONE HEATMAP PER METRIC
# ---------------------------------------------------------
def plot_metric_heatmaps(all_results, metrics, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    for metric in metrics:
        analogy_types, models, mask_positions, matrices = build_metric_matrix(all_results, metric)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        fig.suptitle(f"{metric} Across Analogy Tasks and Models", fontsize=16)

        for i, mp in enumerate(mask_positions):
            im = axes[i].imshow(matrices[mp], cmap="Blues", vmin=0, vmax=1)

            axes[i].set_title(f"Mask Position: {mp}", fontsize=12)
            axes[i].set_xticks(np.arange(len(models)))
            axes[i].set_yticks(np.arange(len(analogy_types)))
            axes[i].set_xticklabels(models, rotation=45, ha="right")
            axes[i].set_yticklabels(analogy_types)
            axes[i].set_xlabel("Model")
            axes[i].set_ylabel("Analogy Type")

            # Annotate cells
            for r in range(len(analogy_types)):
                for c in range(len(models)):
                    axes[i].text(
                        c, r, f"{matrices[mp][r, c]:.2f}",
                        ha="center", va="center", color="black"
                    )

        # Colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)

        plt.savefig(f"{save_dir}/{metric}_heatmap_layout1.png", dpi=300)
        plt.close()

    print("Saved layout 1 heatmaps in:", save_dir)


# ---------------------------------------------------------
# Layout 2: ONE HEATMAP PER MASK POSITION
# ---------------------------------------------------------
def plot_maskposition_heatmaps(all_results, metrics, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    for mp in ["beginning", "middle", "end"]:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 6), constrained_layout=True)
        fig.suptitle(f"Mask Position: {mp}", fontsize=16)

        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            analogy_types, models, mask_positions, matrices = build_metric_matrix(all_results, metric)

            m = matrices[mp]
            ax = axes[idx]
            im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)

            ax.set_title(metric)
            ax.set_xticks(np.arange(len(models)))
            ax.set_yticks(np.arange(len(analogy_types)))
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.set_yticklabels(analogy_types)
            ax.set_xlabel("Model")
            ax.set_ylabel("Analogy Type")

            for r in range(len(analogy_types)):
                for c in range(len(models)):
                    ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center", color="black")

        fig.colorbar(im, ax=axes, shrink=0.6)
        plt.savefig(f"{save_dir}/mask_{mp}_heatmaps_layout2.png", dpi=300)
        plt.close()

    print("Saved layout 2 heatmaps in:", save_dir)


def plot_grouped_bars(all_results, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["Top-1", "Top-10", "Confidence", "Sim"]
    models = ["BERT", "DistilBERT", "RoBERTa", "T5", "GPT-2"]
    analogy_types = sorted(list({x["analogy_type"] for x in all_results}))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.15
        x = np.arange(len(analogy_types))

        for i, model in enumerate(models):
            vals = []
            for at in analogy_types:
                r = [x for x in all_results if x["analogy_type"] == at][0]
                vals.append(r[model][metric])
            ax.bar(x + i * width, vals, width, label=model)

        ax.set_title(f"{metric} — Grouped by Analogy Type")
        ax.set_xticks(x + width*2)
        ax.set_xticklabels(analogy_types, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{metric}_grouped_bar.png", dpi=300)
        plt.close()

    print("Saved grouped bar charts.")

def plot_radar_charts(all_results, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    models = ["BERT", "DistilBERT", "RoBERTa", "T5", "GPT-2"]
    metrics = ["Top-1", "Top-10", "Sim"]

    # Average across all analogy types + mask positions
    import pandas as pd
    df = pd.DataFrame(columns=["Model"] + metrics)

    for model in models:
        rows = []
        for r in all_results:
            rows.append([r[model][m] for m in metrics])
        rows = np.array(rows)
        mean_vals = rows.mean(axis=0)
        df.loc[len(df)] = [model] + mean_vals.tolist()

    # Radar plot for each model
    for _, row in df.iterrows():
        model = row["Model"]
        values = row[metrics].tolist() + [row[metrics[0]]]

        angles = np.linspace(0, 2*np.pi, len(metrics)+1)

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(f"Model Profile: {model}")

        plt.savefig(f"{save_dir}/{model}_radar.png", dpi=300)
        plt.close()

    print("Saved radar charts.")

def plot_lineplots(all_results, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["Top-1", "Top-10", "Confidence", "Sim"]
    models = ["BERT", "DistilBERT", "RoBERTa", "T5", "GPT-2"]
    mask_positions = ["beginning", "middle", "end"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in models:
            vals = []
            for mp in mask_positions:
                rs = [x for x in all_results if x["mask_position"] == mp]
                vals.append(np.mean([xx[model][metric] for xx in rs]))

            ax.plot(mask_positions, vals, marker="o", label=model)

        ax.set_title(f"{metric} vs. Mask Position")
        ax.set_xlabel("Mask Position")
        ax.set_ylabel(metric)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{metric}_lineplot.png", dpi=300)
        plt.close()

    print("Saved line plots.")




# ---------------------------------------------------------
# Master function: call both layouts
# ---------------------------------------------------------
def generate_all_figures(all_results):
    metrics = ["Top-1", "Top-10", "Confidence", "Sim"]

    print("Generating heatmaps (Layout 1)...")
    plot_metric_heatmaps(all_results, metrics)

    print("Generating heatmaps (Layout 2)...")
    plot_maskposition_heatmaps(all_results, metrics)

    print("Generating grouped bar charts...")
    plot_grouped_bars(all_results)

    print("Generating radar charts...")
    plot_radar_charts(all_results)

    print("Generating line plots...")
    plot_lineplots(all_results)


    print("All figures generated.")

