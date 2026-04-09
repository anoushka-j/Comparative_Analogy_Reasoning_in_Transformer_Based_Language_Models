import torch
import numpy as np
from torch.nn.functional import softmax
from transformers import (
    BertTokenizer, BertForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM,
    DistilBertTokenizer, DistilBertForMaskedLM,
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration
)
import re

# -------------------------
# Utilities
# -------------------------
def _clean_token(tok: str) -> str:
    if tok is None: return ""
    tok = tok.replace("##","").replace("Ġ"," ").replace("▁","").replace("\u0120"," ").strip()
    return tok.strip(" \t\n\r'\"`~!@#$%^&*()_+={}[]|\\:;,.<>/?")

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def find_token_index(tokens, target_word):
    """Find first token index that contains the target word (substring match for subwords)"""
    for i, tok in enumerate(tokens):
        if target_word.lower() in tok.lower():
            return i
    return None

def vector_offset_score(hidden, idx_A, idx_B, idx_C, idx_pred):
    """Compute cosine similarity of offset vectors (B-A vs pred-C)"""
    emb = hidden
    offset1 = emb[idx_B] - emb[idx_A]
    offset2 = emb[idx_pred] - emb[idx_C]
    return cosine_similarity(offset1, offset2)

# -------------------------
# Model Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
top_k = 5

# BERT
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).to(device)
bert_model.eval()

# RoBERTa
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True).to(device)
roberta_model.eval()

# DistilBERT
distil_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distil_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased", output_hidden_states=True).to(device)
distil_model.eval()

# GPT-2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
gpt2_model.eval()
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# T5
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small", output_hidden_states=True).to(device)
t5_model.eval()

# -------------------------
# Analogy prompt
# -------------------------
analogy_text = "A lifeguard works at a pool just like a zookeeper works at a [MASK]."
expected_word = "zoo"  # optional reference for evaluation

# -------------------------
# Prediction helpers
# -------------------------
def get_mlm_prediction(model, tokenizer, text, mask_token):
    batch = tokenizer(text.replace("[MASK]", mask_token), return_tensors="pt").to(device)
    mask_idx = (batch["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits[0, mask_idx, :]
    probs = softmax(logits, dim=-1)
    topk = torch.topk(probs, top_k)
    preds = [_clean_token(tokenizer.decode([idx])) for idx in topk.indices[0]]
    hidden = outputs.hidden_states[-1][0].cpu().numpy()
    return preds, hidden, batch

def get_gpt2_prediction(model, tokenizer, text):
    text = text.replace("[MASK]", "").replace("just like", ". Similarly,").rstrip(".!?") + " "
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1]+3,
            num_return_sequences=top_k,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    preds = []
    for out in outputs:
        text_out = tokenizer.decode(out[inputs["input_ids"].shape[1]:]).strip()
        first_phrase = " ".join(text_out.split()[:1]).strip(".,!?")
        if first_phrase:
            preds.append(first_phrase)
    return preds, None, inputs

def get_t5_prediction(model, tokenizer, text):
    prompt = f"Fill in the blank: {text.replace('[MASK]', '___')}"
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**batch, max_new_tokens=2)
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
    candidate = _clean_token(decoded.split()[-1]) if " " in decoded else _clean_token(decoded)
    return [candidate], None, batch

# -------------------------
# Run all models
# -------------------------
results = {}

# BERT
bert_preds, bert_hidden, bert_batch = get_mlm_prediction(bert_model, bert_tokenizer, analogy_text, bert_tokenizer.mask_token)
results["BERT"] = {"preds": bert_preds, "hidden": bert_hidden, "batch": bert_batch}

# RoBERTa
roberta_preds, roberta_hidden, roberta_batch = get_mlm_prediction(roberta_model, roberta_tokenizer, analogy_text, roberta_tokenizer.mask_token)
results["RoBERTa"] = {"preds": roberta_preds, "hidden": roberta_hidden, "batch": roberta_batch}

# DistilBERT
distil_preds, distil_hidden, distil_batch = get_mlm_prediction(distil_model, distil_tokenizer, analogy_text, distil_tokenizer.mask_token)
results["DistilBERT"] = {"preds": distil_preds, "hidden": distil_hidden, "batch": distil_batch}

# GPT-2
gpt2_preds, gpt2_hidden, gpt2_batch = get_gpt2_prediction(gpt2_model, gpt2_tokenizer, analogy_text)
results["GPT-2"] = {"preds": gpt2_preds, "hidden": gpt2_hidden, "batch": gpt2_batch}

# T5
t5_preds, t5_hidden, t5_batch = get_t5_prediction(t5_model, t5_tokenizer, analogy_text)
results["T5"] = {"preds": t5_preds, "hidden": t5_hidden, "batch": t5_batch}

# -------------------------
# Compute vector offsets for MLMs
# -------------------------
for name in ["BERT", "RoBERTa", "DistilBERT"]:
    res = results[name]
    tokens = res["batch"]["input_ids"][0]
    tokens = res["batch"]["input_ids"].new_tensor(tokens)
    tokens = tokenizer.convert_ids_to_tokens(tokens) if name=="BERT" else tokenizer.convert_ids_to_tokens(tokens)
    tokens = res["batch"]["input_ids"][0].cpu()
    tokens = res["batch"]["input_ids"][0]
    tokens = res["batch"]["input_ids"].cpu() if res["batch"] else None
    tokens = res["batch"]["input_ids"][0].cpu().numpy()
    tokens = res["batch"]["input_ids"][0]
    tokens = bert_tokenizer.convert_ids_to_tokens(res["batch"]["input_ids"][0])
    
    # Find indices
    idx_A = find_token_index(tokens, "lifeguard")
    idx_B = find_token_index(tokens, "pool")
    idx_C = find_token_index(tokens, "zoo")
    idx_pred = find_token_index(tokens, _clean_token(res["preds"][0]))
    if None not in [idx_A, idx_B, idx_C, idx_pred]:
        sim = vector_offset_score(res["hidden"], idx_A, idx_B, idx_C, idx_pred)
        res["offset_cosine"] = sim
    else:
        res["offset_cosine"] = None

# -------------------------
# Print table
# -------------------------
print("\n=== Analogy Comparison Table ===")
print(f"{'Model':<12} {'Top Prediction':<15} {'Offset Cosine':<12}")
for name, res in results.items():
    top_pred = res["preds"][0] if res["preds"] else "N/A"
    offset = f"{res['offset_cosine']:.3f}" if res.get("offset_cosine") else "N/A"
    print(f"{name:<12} {top_pred:<15} {offset:<12}")
