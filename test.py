import torch
from torch.nn.functional import softmax, cosine_similarity
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import string

# ================================
# BERT Setup
# ================================
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
bert_model.eval()

# Analogy prompt for BERT
bert_text = "A pilot works for an airline, just like a sailor works for a [MASK]."

# Tokenize and get mask index
bert_inputs = bert_tokenizer(bert_text, return_tensors="pt")
mask_idx = (bert_inputs["input_ids"] == bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

# Run BERT
with torch.no_grad():
    bert_outputs = bert_model(**bert_inputs)

# Get top predictions
logits = bert_outputs.logits[0, mask_idx, :]
probs = softmax(logits, dim=-1)
top_k = 10
topk = torch.topk(probs, top_k)

# Decode and filter punctuation
bert_preds, bert_probs = [], []
for idx, prob in zip(topk.indices[0], topk.values[0]):
    word = bert_tokenizer.decode([idx]).strip()
    if all(c not in string.punctuation for c in word):
        bert_preds.append(word)
        bert_probs.append(float(prob))

# ================================
# Compute vector offsets
# ================================
# Extract hidden states of last layer
last_hidden = bert_outputs.hidden_states[-1][0]  # shape: (seq_len, hidden_dim)
tokens = bert_tokenizer.convert_ids_to_tokens(bert_inputs["input_ids"][0])

# Helper to get embedding of a token (first occurrence)
def get_token_vector(token_str):
    try:
        idx = tokens.index(token_str)
        return last_hidden[idx]
    except ValueError:
        return None

# Example: pilot → airline vs sailor → top prediction
vec_pilot = get_token_vector("pilot")
vec_airline = get_token_vector("airline")
vec_sailor = get_token_vector("sailor")
vec_pred = get_token_vector(bert_preds[0]) if bert_preds else None

if None not in [vec_pilot, vec_airline, vec_sailor, vec_pred]:
    offset1 = vec_airline - vec_pilot
    offset2 = vec_pred - vec_sailor
    cos_sim = cosine_similarity(offset1.unsqueeze(0), offset2.unsqueeze(0)).item()
    print(f"\nCosine similarity of analogy vectors (pilot→airline vs sailor→{bert_preds[0]}): {cos_sim:.4f}")
else:
    print("\nCould not compute vector offsets: some tokens not found in tokenized input.")

# ================================
# GPT-2 Setup
# ================================
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Analogy prompt for GPT-2 (more natural phrasing)
gpt2_text = "A pilot works for an airline. Similarly, a sailor works for a"
gpt2_inputs = gpt2_tokenizer(gpt2_text, return_tensors="pt")

with torch.no_grad():
    gpt2_outputs = gpt2_model.generate(
        **gpt2_inputs,
        max_length=gpt2_inputs["input_ids"].shape[1] + 3,
        num_return_sequences=10,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )

gpt2_preds = []
for out in gpt2_outputs:
    text_out = gpt2_tokenizer.decode(out[gpt2_inputs["input_ids"].shape[1]:]).strip()
    first_phrase = " ".join(text_out.split()[:1])
    first_phrase = first_phrase.strip(string.punctuation)
    gpt2_preds.append(first_phrase)

# ================================
# Print Results
# ================================
print("\n=== BERT Predictions ===")
for word, prob in zip(bert_preds, bert_probs):
    print(f"{word:15s}  {prob:.4f}")

print("\n=== GPT-2 Predictions ===")
for pred in gpt2_preds:
    print(pred)
