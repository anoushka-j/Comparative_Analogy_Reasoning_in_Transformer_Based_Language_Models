import streamlit as st
import torch
import string
from torch.nn.functional import softmax
from transformers import (
    BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel,
    RobertaTokenizer, RobertaForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM
)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import re

# -------------------------
# Original BERT + GPT-2 code (kept exactly as-is)
# -------------------------
@st.cache_resource
def load_models():
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    bert_model.eval()

    # GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()

    # Add padding token if missing
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    return bert_tokenizer, bert_model, gpt2_tokenizer, gpt2_model

bert_tokenizer, bert_model, gpt2_tokenizer, gpt2_model = load_models()

st.title("Analogy Reasoning Dashboard 🧠")
st.write("Compare BERT and GPT-2 predictions for analogies.")

analogy_input = st.text_input("Enter an analogy:", "A pilot works for an airline, just like a sailor works for a [MASK].")
top_k = st.slider("Top K predictions for BERT:", 1, 20, 10)

def get_bert_predictions(text, top_k=10):
    bert_inputs = bert_tokenizer(text, return_tensors="pt")
    mask_idx = (bert_inputs["input_ids"] == bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    
    logits = bert_outputs.logits[0, mask_idx, :]
    probs = softmax(logits, dim=-1)
    topk = torch.topk(probs, top_k)

    bert_preds, bert_probs = [], []
    for idx, prob in zip(topk.indices[0], topk.values[0]):
        word = bert_tokenizer.decode([idx]).strip()
        if all(c not in string.punctuation for c in word):
            bert_preds.append(word)
            bert_probs.append(float(prob))
    
    return bert_preds, bert_probs

def get_gpt2_predictions(text, num_return_sequences=10):
    text="A pilot works for an airline. Similarly, a sailor works for a"

    gpt2_inputs = gpt2_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        gpt2_outputs = gpt2_model.generate(
            **gpt2_inputs,
            max_length=gpt2_inputs["input_ids"].shape[1] + 3,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )

    gpt2_preds = []
    for out in gpt2_outputs:
        text_out = gpt2_tokenizer.decode(out[gpt2_inputs["input_ids"].shape[1]:]).strip()
        first_phrase = " ".join(text_out.split()[:1])
        first_phrase = first_phrase.replace(gpt2_tokenizer.eos_token, "").strip()
        gpt2_preds.append(first_phrase)
    
    return gpt2_preds

if "[MASK]" in analogy_input:
    bert_preds, bert_probs = get_bert_predictions(analogy_input, top_k=top_k)
    st.subheader("BERT Predictions")
    for word, prob in zip(bert_preds, bert_probs):
        st.write(f"{word:15s}  {prob:.4f}")
else:
    st.info("Add a `[MASK]` token for BERT to predict.")

gpt2_input_text = analogy_input.replace("[MASK]", "")
gpt2_preds = get_gpt2_predictions(gpt2_input_text)
st.subheader("GPT-2 Predictions")
for pred in gpt2_preds:
    st.write(pred)

# -------------------------
# New: RoBERTa + DistilBERT with full dashboard visuals
# -------------------------

def _clean_token(tok: str) -> str:
    if tok is None:
        return ""
    tok = tok.replace("##", "").replace("Ġ", " ").replace("▁", "").replace("\u0120", " ")
    tok = tok.strip()
    return tok.strip(" \t\n\r'\"`~!@#$%^&*()_+={}[]|\\:;,.<>/?")

def _is_reasonable_word(tok: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z'-]*$", tok))

@st.cache_resource
def load_masked_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_model = RobertaForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True).to(device)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased", output_hidden_states=True).to(device)
    return device, roberta_tokenizer, roberta_model, distilbert_tokenizer, distilbert_model

device, roberta_tokenizer, roberta_model, distilbert_tokenizer, distilbert_model = load_masked_models()

def get_masked_predictions(tokenizer, model, text, mask_token, top_k=10):
    masked_text = text.replace("[MASK]", mask_token)
    inputs = tokenizer(masked_text, return_tensors="pt").to(device)
    mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, mask_idx, :]
    probs = softmax(logits, dim=-1)
    topk_vals = torch.topk(probs, top_k)
    preds = []
    for idx in topk_vals.indices[0]:
        tok = _clean_token(tokenizer.decode([idx]))
        if _is_reasonable_word(tok):
            preds.append(tok)
    hidden = [h.detach().cpu() for h in outputs.hidden_states] if outputs.hidden_states is not None else None
    return preds, hidden, inputs

def plot_confidences(preds, model):
    fig = px.bar(x=preds, y=[1.0/len(preds)]*len(preds), title=f"Predictions ({model})",
                 labels={'x': 'Prediction', 'y': 'Confidence'})
    st.plotly_chart(fig, use_container_width=True)

def plot_embeddings(hidden_states, tokenizer, inputs, model):
    if hidden_states is None:
        st.info(f"{model} embeddings not available")
        return
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    embeddings = hidden_states[-1][0].numpy()
    reducer = PCA(n_components=2) if len(tokens)<3 else TSNE(n_components=2, random_state=42, perplexity=min(30, max(2,len(tokens)-1)))
    reduced = reducer.fit_transform(embeddings)
    fig = px.scatter(x=reduced[:,0], y=reduced[:,1], text=tokens, title=f"Token Embeddings ({model})")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# Run dashboard for masked LMs
st.subheader("RoBERTa Predictions")
roberta_preds, roberta_hidden, roberta_inputs = get_masked_predictions(roberta_tokenizer, roberta_model, analogy_input, "<mask>", top_k)
st.write("**Top Predictions:**", roberta_preds)
plot_confidences(roberta_preds, "RoBERTa")
plot_embeddings(roberta_hidden, roberta_tokenizer, roberta_inputs, "RoBERTa")

st.subheader("DistilBERT Predictions")
distil_preds, distil_hidden, distil_inputs = get_masked_predictions(distilbert_tokenizer, distilbert_model, analogy_input, "[MASK]", top_k)
st.write("**Top Predictions:**", distil_preds)
plot_confidences(distil_preds, "DistilBERT")
plot_embeddings(distil_hidden, distilbert_tokenizer, distil_inputs, "DistilBERT")
