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
from visualization import generate_all_figures
# ============================================================
# Import benchmark dataset
# ============================================================
from dataset import benchmark  # list of dicts with fields: analogy_type, mask_position, correct_answers, prompt_bert, prompt_roberta, prompt_t5, prompt_gpt


# ============================================================
# Helper functions
# ============================================================

def semantic_similarity(model, tokenizer, word, correct_answers):
    """Cosine similarity between a predicted word and the closest correct answer."""
    sims = []
    for ans in correct_answers:
        print(ans)
        inputs = tokenizer([word, ans], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1)
        sim = F.cosine_similarity(emb[0].unsqueeze(0), emb[1].unsqueeze(0))
        sims.append(sim.item())
    return max(sims)


def diversity(preds, correct_answers):
    """Fraction of predictions that are correct among top-k."""
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
# Evaluation loop
# ============================================================
all_results = []

for example in benchmark:
    correct_answers = example.get("correct_answers")
    analogy_type = example["analogy_type"]
    mask_position = example["mask_position"]

    # ---------------- BERT ----------------
    bert_text = example["bert_prompt"]
    #print(bert_text)
    bert_inputs = bert_tokenizer(bert_text, return_tensors="pt")
    mask_idx = (bert_inputs["input_ids"] == bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    with torch.no_grad():
        outputs = bert_model(**bert_inputs)
    logits = outputs.logits[0, mask_idx, :]
    probs = softmax(logits, dim=-1)
    topk = torch.topk(probs, 10)
    bert_preds = [bert_tokenizer.decode([idx]).strip() for idx in topk.indices[0] if all(c not in string.punctuation for c in bert_tokenizer.decode([idx]).strip())]
    bert_confs = [float(probs[0, idx]) for idx in topk.indices[0]]

    # ---------------- DistilBERT ----------------
    distil_text = example["bert_prompt"]  # same prompt as BERT
    distil_inputs = distil_tokenizer(distil_text, return_tensors="pt")
    distil_mask = (distil_inputs["input_ids"] == distil_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    with torch.no_grad():
        outputs = distil_model(**distil_inputs)
    logits = outputs.logits[0, distil_mask, :]
    probs = softmax(logits, dim=-1)
    topk = torch.topk(probs, 10)
    distil_preds = [distil_tokenizer.decode([idx]).strip() for idx in topk.indices[0] if all(c not in string.punctuation for c in distil_tokenizer.decode([idx]).strip())]
    distil_confs = [float(probs[0, idx]) for idx in topk.indices[0]]

    # ---------------- RoBERTa ----------------
    roberta_text = example["roberta_prompt"]
    roberta_inputs = roberta_tokenizer(roberta_text, return_tensors="pt")
    roberta_mask = (roberta_inputs["input_ids"] == roberta_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    with torch.no_grad():
        outputs = roberta_model(**roberta_inputs)
    logits = outputs.logits[0, roberta_mask, :]
    probs = softmax(logits, dim=-1)
    topk = torch.topk(probs, 10)
    roberta_preds = [roberta_tokenizer.decode([idx]).strip() for idx in topk.indices[0] if all(c not in string.punctuation for c in roberta_tokenizer.decode([idx]).strip())]
    roberta_confs = [float(probs[0, idx]) for idx in topk.indices[0]]

    # ==== PRINT PROMPT + MASKED-LM PREDICTIONS ====
    print("\n==============================================")
    print(f"Analogy type: {analogy_type} | Mask: {mask_position}")
    print("Correct answers:", correct_answers)
    print("\nBERT prompt:     ", bert_text)
    print("BERT preds:      ", bert_preds[:10])

    print("\nDistilBERT prompt:", distil_text)
    print("DistilBERT preds:", distil_preds[:10])

    print("\nRoBERTa prompt:  ", roberta_text)
    print("RoBERTa preds:   ", roberta_preds[:10])
    print("==============================================\n")

    # ---------------- T5 ----------------
    t5_text = example["t5_prompt"]
    print(t5_text)
    t5_inputs = t5_tokenizer(t5_text, return_tensors="pt")
    k = 10

    with torch.no_grad():
        t5_out = t5_model.generate(
            **t5_inputs,
            num_beams=k,
            num_return_sequences=k,
            max_length=5,
            early_stopping=True
        )

    t5_preds = []
    t5_confs = []

    for seq in t5_out:
        decoded = t5_tokenizer.decode(seq, skip_special_tokens=True)
        # Remove placeholder and split into words
        words = decoded.replace("<extra_id_0>", "").strip().split()
        
        # Make sure there are at least 2 words
        if len(words) < 2:
            continue

        # Take the second word
        pred_word = words[1]

        # Skip duplicates
        if pred_word in t5_preds:
            continue

        # Confidence for the second token in the sequence
        second_token_id = seq[2].item()  # seq[0] = <s>, seq[1] = first token
        logits = t5_model(**t5_inputs, labels=seq.unsqueeze(0)).logits[0, 2]
        conf = float(F.softmax(logits, dim=-1)[second_token_id])
        #print(pred_word)
        t5_preds.append(pred_word)
        print(pred_word)
        t5_confs.append(conf)
    
    # ---------------- GPT-2 ----------------
    gpt_text = example["gpt2_prompt"]
    #print(gpt_text)
    gpt_inputs = gpt2_tokenizer(gpt_text, return_tensors="pt")
    with torch.no_grad():
        logits = gpt2_model(**gpt_inputs).logits
        next_logits = logits[0, -1]
        next_probs = softmax(next_logits, dim=-1)
    _, top_indices = torch.topk(next_probs, 10)
    gpt2_preds = [gpt2_tokenizer.decode([idx]).strip(string.punctuation) for idx in top_indices]
    gpt2_confs = [float(next_probs[idx]) for idx in top_indices]

    # ---------------- Metrics ----------------
    results = {
        "analogy_type": analogy_type,
        "mask_position": mask_position,
        "BERT": {"Top-1": top1_accuracy(bert_preds, correct_answers),
                 "Top-10": topk_accuracy(bert_preds, correct_answers),
                 "Confidence": bert_confs[0] if bert_confs else 0.0,
                 "Sim": semantic_similarity(sim_model, sim_tokenizer, bert_preds[0], correct_answers) if bert_preds else 0.0,
                 "Diversity": diversity(bert_preds, correct_answers)},
        "DistilBERT": {"Top-1": top1_accuracy(distil_preds, correct_answers),
                       "Top-10": topk_accuracy(distil_preds, correct_answers),
                       "Confidence": distil_confs[0] if distil_confs else 0.0,
                       "Sim": semantic_similarity(sim_model, sim_tokenizer, distil_preds[0], correct_answers) if distil_preds else 0.0,
                       "Diversity": diversity(distil_preds, correct_answers)},
        "RoBERTa": {"Top-1": top1_accuracy(roberta_preds, correct_answers),
                    "Top-10": topk_accuracy(roberta_preds, correct_answers),
                    "Confidence": roberta_confs[0] if roberta_confs else 0.0,
                    "Sim": semantic_similarity(sim_model, sim_tokenizer, roberta_preds[0], correct_answers) if roberta_preds else 0.0,
                    "Diversity": diversity(roberta_preds, correct_answers)},
        "T5": {"Top-1": top1_accuracy(t5_preds, correct_answers),
               "Top-10": topk_accuracy(t5_preds, correct_answers),
               "Confidence": t5_confs[0] if t5_confs else 0.0,
               "Sim": semantic_similarity(sim_model, sim_tokenizer, t5_preds[0], correct_answers) if t5_preds else 0.0,
               "Diversity": diversity(t5_preds, correct_answers)},
        "GPT-2": {"Top-1": top1_accuracy(gpt2_preds, correct_answers),
                  "Top-10": topk_accuracy(gpt2_preds, correct_answers),
                  "Confidence": gpt2_confs[0] if gpt2_confs else 0.0,
                  "Sim": semantic_similarity(sim_model, sim_tokenizer, gpt2_preds[0], correct_answers) if gpt2_preds else 0.0,
                  "Diversity": diversity(gpt2_preds, correct_answers)}
    }

    all_results.append(results)

# ============================================================
# Print summary
# ============================================================
for r in all_results:
    print(f"\nAnalogy type: {r['analogy_type']} | Mask: {r['mask_position']}")
    for model in ["BERT", "DistilBERT", "RoBERTa", "T5", "GPT-2"]:
        metrics = r[model]
        print(f"{model:10s} | Top-1: {metrics['Top-1']:.2f} | Top-10: {metrics['Top-10']:.2f} | Conf: {metrics['Confidence']:.4f} | Sim: {metrics['Sim']:.4f} | Div: {metrics['Diversity']:.2f}")

import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# AGGREGATE RESULTS BY (analogy_type, mask_position)
# ============================================================
grouped = defaultdict(lambda: defaultdict(list))

for r in all_results:
    grouped[r["analogy_type"]][r["mask_position"]].append(r)


def mean(values):
    return sum(values) / len(values) if values else 0.0


# ============================================================
# COMPUTE AGGREGATED METRICS
# ============================================================
aggregated = {}

for analogy_type, positions in grouped.items():
    aggregated[analogy_type] = {}
    for mask_pos, examples in positions.items():
        aggregated[analogy_type][mask_pos] = {}

        for model in ["BERT", "DistilBERT", "RoBERTa", "T5", "GPT-2"]:
            metrics = {"Top-1": [], "Top-10": [], "Confidence": [], "Sim": [], "Diversity": []}

            for ex in examples:
                for m in metrics:
                    metrics[m].append(ex[model][m])

            aggregated[analogy_type][mask_pos][model] = {m: mean(v) for m, v in metrics.items()}


# ============================================================
# PRINT AGGREGATED SUMMARY
# ============================================================
print("\n\n==================== AGGREGATED METRICS ====================")
for analogy_type, positions in aggregated.items():
    for mask_pos, models in positions.items():
        print(f"\n### Analogy: {analogy_type} | Mask: {mask_pos}")
        for model, m in models.items():
            print(f"{model:10} | Top-1={m['Top-1']:.3f} | Top-10={m['Top-10']:.3f} "
                  f"| Conf={m['Confidence']:.4f} | Sim={m['Sim']:.4f} | Div={m['Diversity']:.3f}")


# ============================================================
# PLOTTING FUNCTION
# ============================================================
def plot_metric(metric_name):
    """
    Creates one plot per analogy type × mask position.
    Each plot contains performance bars for all models.
    """
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
# GENERATE ALL METRIC PLOTS
# ============================================================
generate_all_figures(all_results)