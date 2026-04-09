import torch
from torch.nn.functional import softmax
from transformers import (
    BertTokenizer, BertForMaskedLM,
    DistilBertTokenizer, DistilBertForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM,
    T5Tokenizer, T5ForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModel
)
import torch.nn.functional as F
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import umap

# ============================================================
# Import benchmark dataset
# ============================================================
from dataset import benchmark  # list of dicts with fields: analogy_type, mask_position, correct_answers, prompt_bert, prompt_roberta, prompt_t5, prompt_gpt

# ============================================================
# Global constants
# ============================================================
DEFAULT_CORRECT = ["ship", "boat", "vessel"]
save_dir = "figures"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================
def semantic_similarity(model, tokenizer, word, correct_answers):
    sims = []
    for ans in correct_answers:
        inputs = tokenizer([word, ans], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1)
        sim = F.cosine_similarity(emb[0].unsqueeze(0), emb[1].unsqueeze(0))
        sims.append(sim.item())
    return max(sims)

def diversity(preds, correct_answers):
    valid = sum(1 for p in preds if p.lower().strip() in correct_answers)
    return valid / len(preds) if preds else 0.0

def top1_accuracy(preds, correct_answers):
    return 1.0 if preds and preds[0].lower().strip() in correct_answers else 0.0

def topk_accuracy(preds, correct_answers):
    return 1.0 if any(p.lower().strip() in correct_answers for p in preds) else 0.0

# ============================================================
# Load semantic similarity model
# ============================================================
sim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sim_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# ============================================================
# Load NLP models
# ============================================================
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
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# ============================================================
# Setup embedding storage for PCA/UMAP
# ============================================================
embedding_store = {model: {"embeddings": [], "analogy_type": [], "mask_position": []}
                   for model in ["BERT", "DistilBERT", "RoBERTa", "T5", "GPT-2"]}

prompt_keys = {
    "BERT": "bert_prompt",
    "DistilBERT": "distil_prompt",
    "RoBERTa": "roberta_prompt",
    "T5": "t5_prompt",
    "GPT-2": "gpt2_prompt",
}

models_and_tokenizers = {
    "BERT": (bert_tokenizer, bert_model, False, False),
    "DistilBERT": (distil_tokenizer, distil_model, False, False),
    "RoBERTa": (roberta_tokenizer, roberta_model, False, False),
    "T5": (t5_tokenizer, t5_model, True, False),
    "GPT-2": (gpt2_tokenizer, gpt2_model, False, True),
}

# ============================================================
# Evaluation loop
# ============================================================
all_results = []

for example in benchmark:
    correct_answers = example.get("correct_answers", DEFAULT_CORRECT)
    analogy_type = example["analogy_type"]
    mask_position = example["mask_position"]

    # ---------------- PROCESS EACH MODEL ----------------
    model_results = {}
    for model_name, (tokenizer, model, is_t5, is_gpt2) in models_and_tokenizers.items():
        text = example[prompt_keys[model_name]]

        # Get predictions
        if model_name in ["BERT", "DistilBERT", "RoBERTa"]:
            inputs = tokenizer(text, return_tensors="pt")
            mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, mask_idx, :]
            probs = softmax(logits, dim=-1)
            topk = torch.topk(probs, 10)
            preds = [tokenizer.decode([idx]).strip() for idx in topk.indices[0]
                     if all(c not in string.punctuation for c in tokenizer.decode([idx]).strip())]
            confs = [float(probs[0, idx]) for idx in topk.indices[0]]

        elif model_name == "T5":
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                t5_out = model.generate(**inputs, num_beams=10, num_return_sequences=10, max_length=5, early_stopping=True)
            preds, confs = [], []
            for seq in t5_out:
                decoded = tokenizer.decode(seq)
                pred = decoded.replace("<pad>", "").replace("</s>", "").replace("<extra_id_0>", "").replace("<extra_id_1>", "").strip()
                gen_ids = seq
                first_token = gen_ids[0].item()
                with torch.no_grad():
                    logits = model(**inputs, labels=gen_ids.unsqueeze(0)).logits[0, 0]
                    conf = float(F.softmax(logits, dim=-1)[first_token])
                if pred not in preds:
                    preds.append(pred)
                    confs.append(conf)

        else:  # GPT-2
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                next_logits = logits[0, -1]
                next_probs = softmax(next_logits, dim=-1)
            _, top_indices = torch.topk(next_probs, 10)
            preds = [tokenizer.decode([idx]).strip(string.punctuation) for idx in top_indices]
            confs = [float(next_probs[idx]) for idx in top_indices]

        # ---------------- METRICS ----------------
        model_results[model_name] = {
            "Top-1": top1_accuracy(preds, correct_answers),
            "Top-10": topk_accuracy(preds, correct_answers),
            "Confidence": confs[0] if confs else 0.0,
            "Sim": semantic_similarity(sim_model, sim_tokenizer, preds[0], correct_answers) if preds else 0.0,
            "Diversity": diversity(preds, correct_answers)
        }

        # ---------------- EMBEDDING EXTRACTION ----------------
        with torch.no_grad():
            if is_t5:
                emb = model.encoder(**inputs).last_hidden_state
            else:
                emb = model(**inputs).last_hidden_state
        mean_emb = emb.mean(dim=1).squeeze().numpy()
        embedding_store[model_name]["embeddings"].append(mean_emb)
        embedding_store[model_name]["analogy_type"].append(analogy_type)
        embedding_store[model_name]["mask_position"].append(mask_position)

    # Save results for this example
    results = {"analogy_type": analogy_type, "mask_position": mask_position, **model_results}
    all_results.append(results)

# ============================================================
# PRINT SUMMARY
# ============================================================
for r in all_results:
    print(f"\nAnalogy type: {r['analogy_type']} | Mask: {r['mask_position']}")
    for model in models_and_tokenizers.keys():
        metrics = r[model]
        print(f"{model:10s} | Top-1: {metrics['Top-1']:.2f} | Top-10: {metrics['Top-10']:.2f} | "
              f"Conf: {metrics['Confidence']:.4f} | Sim: {metrics['Sim']:.4f} | Div: {metrics['Diversity']:.2f}")

# ============================================================
# AGGREGATE RESULTS BY (analogy_type, mask_position)
# ============================================================
grouped = defaultdict(lambda: defaultdict(list))
for r in all_results:
    grouped[r["analogy_type"]][r["mask_position"]].append(r)

def mean(values):
    return sum(values) / len(values) if values else 0.0

aggregated = {}
for analogy_type, positions in grouped.items():
    aggregated[analogy_type] = {}
    for mask_pos, examples in positions.items():
        aggregated[analogy_type][mask_pos] = {}
        for model in models_and_tokenizers.keys():
            metrics = {"Top-1": [], "Top-10": [], "Confidence": [], "Sim": [], "Diversity": []}
            for ex in examples:
                for m in metrics:
                    metrics[m].append(ex[model][m])
            aggregated[analogy_type][mask_pos][model] = {m: mean(v) for m, v in metrics.items()}

# ============================================================
# PLOTTING FUNCTION FOR METRICS
# ============================================================
def plot_metric(metric_name):
    for analogy_type, positions in aggregated.items():
        for mask_pos, models in positions.items():
            model_names = list(models.keys())
            metric_values = [models[m][metric_name] for m in model_names]
            plt.figure(figsize=(8, 5))
            plt.bar(model_names, metric_values)
            plt.title(f"{metric_name} — {analogy_type} | Mask: {mask_pos}")
            plt.xlabel("Models")
            plt.ylabel(metric_name)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# ============================================================
# CALL EXISTING FIGURES (HEATMAPS, RADAR, LINE PLOTS)
# ============================================================
from visualization import generate_all_figures
generate_all_figures(all_results)

# ============================================================
# PCA / UMAP EMBEDDING VISUALIZATIONS
# ============================================================
for model_name, data in embedding_store.items():
    embeddings = np.array(data["embeddings"])
    mask_positions = np.array(data["mask_position"])
    if embeddings.shape[0] == 0:
        continue

    # PCA
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(embeddings)

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_proj = reducer.fit_transform(embeddings)

    for proj, method in zip([pca_proj, umap_proj], ["PCA", "UMAP"]):
        plt.figure(figsize=(8, 8))
        for mp in np.unique(mask_positions):
            idxs = np.where(mask_positions == mp)[0]
            plt.scatter(proj[idxs, 0], proj[idxs, 1], alpha=0.7, label=f"{mp}", s=50)
        plt.title(f"{method} Embeddings — {model_name} (colored by mask position)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name}_{method}_mask.png", dpi=300)
        plt.close()

print("All PCA and UMAP embedding plots saved.")
