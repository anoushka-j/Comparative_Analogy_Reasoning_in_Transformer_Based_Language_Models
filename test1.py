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
import string
import torch.nn.functional as F

# ============================================================
# Helper functions
# ============================================================

def semantic_similarity(model, tokenizer, w1, w2):
    """Cosine similarity between embeddings of two words."""
    inputs = tokenizer([w1, w2], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1)
    sim = F.cosine_similarity(emb[0].unsqueeze(0), emb[1].unsqueeze(0))
    return float(sim.item())

def diversity(preds):
    """Unique predictions / total predictions."""
    if not preds: 
        return 0.0
    return len(set(preds)) / len(preds)

# Embedding model for semantic similarity
sim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sim_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# ============================================================
# BERT
# ============================================================
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_model.eval()

bert_text = "A pilot works for an airline, just like a sailor works for a [MASK]."
bert_inputs = bert_tokenizer(bert_text, return_tensors="pt")
mask_idx = (bert_inputs["input_ids"] == bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

with torch.no_grad():
    bert_outputs = bert_model(**bert_inputs)

logits = bert_outputs.logits[0, mask_idx, :]
probs = softmax(logits, dim=-1)
topk = torch.topk(probs, 10)

bert_preds = []
bert_confs = []
for idx in topk.indices[0]:
    word = bert_tokenizer.decode([idx]).strip()
    conf = float(probs[0, idx])
    if all(c not in string.punctuation for c in word):
        bert_preds.append(word)
        bert_confs.append(conf)

# ============================================================
# DistilBERT
# ============================================================
distil_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distil_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
distil_model.eval()

distil_text = "A pilot works for an airline, just like a sailor works for a [MASK]."
distil_inputs = distil_tokenizer(distil_text, return_tensors="pt")
distil_mask = (distil_inputs["input_ids"] == distil_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

with torch.no_grad():
    distil_out = distil_model(**distil_inputs)

distil_logits = distil_out.logits[0, distil_mask, :]
distil_probs = softmax(distil_logits, dim=-1)
distil_topk = torch.topk(distil_probs, 10)

distil_preds = []
distil_confs = []
for idx in distil_topk.indices[0]:
    word = distil_tokenizer.decode([idx]).strip()
    conf = float(distil_probs[0, idx])
    if all(c not in string.punctuation for c in word):
        distil_preds.append(word)
        distil_confs.append(conf)

# ============================================================
# RoBERTa
# ============================================================
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForMaskedLM.from_pretrained("roberta-base")
roberta_model.eval()

roberta_text = "A pilot works for an airline, just like a sailor works for a <mask>."
roberta_inputs = roberta_tokenizer(roberta_text, return_tensors="pt")
roberta_mask = (roberta_inputs["input_ids"] == roberta_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

with torch.no_grad():
    roberta_out = roberta_model(**roberta_inputs)

roberta_logits = roberta_out.logits[0, roberta_mask, :]
roberta_probs = softmax(roberta_logits, dim=-1)
roberta_topk = torch.topk(roberta_probs, 10)

roberta_preds = []
roberta_confs = []
for idx in roberta_topk.indices[0]:
    word = roberta_tokenizer.decode([idx]).strip()
    conf = float(roberta_probs[0, idx])
    if all(c not in string.punctuation for c in word):
        roberta_preds.append(word)
        roberta_confs.append(conf)

# ============================================================
# T5
# ============================================================
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_model.eval()

t5_text = "Fill in the blank: A pilot works for an airline, just like a sailor works for a <extra_id_0>."
t5_inputs = t5_tokenizer(t5_text, return_tensors="pt")

with torch.no_grad():
    t5_out = t5_model.generate(**t5_inputs, num_beams=5, max_length=10)

decoded = t5_tokenizer.decode(t5_out[0])
t5_pred = decoded.replace("<extra_id_0>", "").replace("<extra_id_1>", "").strip()

# T5 confidence: probability of its first generated token
gen_ids = t5_out[0]
first_token = gen_ids[0].item()

with torch.no_grad():
    logits = t5_model(**t5_inputs, labels=gen_ids.unsqueeze(0)).logits[0, 0]
    t5_conf = float(softmax(logits, dim=-1)[first_token])

# ============================================================
# GPT-2
# ============================================================
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_text = "A pilot works for an airline. Similarly, a sailor works for a"
gpt2_inputs = gpt2_tokenizer(gpt2_text, return_tensors="pt")

with torch.no_grad():
    logits = gpt2_model(**gpt2_inputs).logits
    next_logits = logits[0, -1]           # distribution for next token
    next_probs = softmax(next_logits, dim=-1)

_, top_indices = torch.topk(next_probs, 10)

gpt2_preds = []
gpt2_confs = []
for idx in top_indices:
    word = gpt2_tokenizer.decode([idx]).strip(string.punctuation)
    gpt2_preds.append(word)
    gpt2_confs.append(float(next_probs[idx]))

# ============================================================
# METRICS
# ============================================================

correct_answer = ["ship"]

def top1_accuracy(preds): return 1.0 if preds and preds[0].lower().strip() == correct_answer else 0.0
def topk_accuracy(preds): return 1.0 if correct_answer in [p.lower() for p in preds] else 0.0

# Semantic similarity
def sim(pred): return semantic_similarity(sim_model, sim_tokenizer, pred, correct_answer)

eval_results = {
    "Model":             ["BERT", "DistilBERT", "RoBERTa", "T5", "GPT-2"],
    "Top-1 Accuracy":    [top1_accuracy(bert_preds), top1_accuracy(distil_preds),
                          top1_accuracy(roberta_preds), top1_accuracy([t5_pred]),
                          top1_accuracy(gpt2_preds)],
    "Top-10 Accuracy":   [topk_accuracy(bert_preds), topk_accuracy(distil_preds),
                          topk_accuracy(roberta_preds), topk_accuracy([t5_pred]),
                          topk_accuracy(gpt2_preds)],
    "Confidence Score":  [bert_confs[0], distil_confs[0],
                          roberta_confs[0], t5_conf, gpt2_confs[0]],
    "Semantic Similarity":[sim(bert_preds[0]), sim(distil_preds[0]),
                           sim(roberta_preds[0]), sim(t5_pred),
                           sim(gpt2_preds[0])],
    "Diversity":         [diversity(bert_preds), diversity(distil_preds),
                          diversity(roberta_preds), 0.0,
                          diversity(gpt2_preds)]
}

# ============================================================
# PRINT TABLE
# ============================================================

print("\n=== MODEL EVALUATION RESULTS ===")
for i in range(5):
    print(
        f"{eval_results['Model'][i]:10s} | "
        f"Top-1: {eval_results['Top-1 Accuracy'][i]:.2f} | "
        f"Top-10: {eval_results['Top-10 Accuracy'][i]:.2f} | "
        f"Conf: {eval_results['Confidence Score'][i]:.4f} | "
        f"Sim: {eval_results['Semantic Similarity'][i]:.4f} | "
        f"Div: {eval_results['Diversity'][i]:.2f}"
    )

print(roberta_preds)
print(gpt2_preds)
print(bert_preds)