# app_dashboard.py
# Updated full file with cleaned token handling and T5 single-word output mode (A)
#
# Requirements:
# pip install streamlit torch transformers sentencepiece matplotlib seaborn scikit-learn plotly

import streamlit as st
import string
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.nn.functional import softmax
from transformers import (
    BertTokenizer, BertForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM,
    DistilBertTokenizer, DistilBertForMaskedLM,
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration
)
import re
import plotly.express as px
import numpy as np

# -------------------------
# Utility helpers
# -------------------------
def _clean_token_static(tok: str) -> str:
    if tok is None:
        return ""

    # Common cleanup but SAFELY
    tok = tok.replace("##", "")
    tok = tok.replace("Ġ", " ")
    tok = tok.replace("▁", "")
    tok = tok.replace("\u0120", " ")

    tok = tok.strip()

    # DO NOT STRIP REAL LETTERS – only kill leading/trailing punctuation
    tok = tok.strip(" \t\n\r'\"`~!@#$%^&*()_+={}[]|\\:;,.<>/?")

    return tok


def _is_reasonable_word(tok: str) -> bool:
    # allow single letters, allow lowercase, uppercase, hyphen, apostrophe
    return bool(re.match(r"^[A-Za-z][A-Za-z'-]*$", tok))

# -------------------------
# Analogy Evaluator
# -------------------------
class AnalogyEvaluator:
    def __init__(self, top_k=10):
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load models once (may take time + memory)
        # consider reducing model sizes if you run into memory issues
        self.models = {
            "BERT": {
                "tokenizer": BertTokenizer.from_pretrained("bert-base-uncased"),
                "model": BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).to(self.device),
                "mask_token": "[MASK]"
            },
            "RoBERTa": {
                "tokenizer": RobertaTokenizer.from_pretrained("roberta-base"),
                "model": RobertaForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True).to(self.device),
                "mask_token": "<mask>"
            },
            "DistilBERT": {
                "tokenizer": DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
                "model": DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased", output_hidden_states=True).to(self.device),
                "mask_token": "[MASK]"
            },
            "GPT-2": {
                "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
                "model": GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(self.device),
            },
            "T5": {
                "tokenizer": T5Tokenizer.from_pretrained("t5-small"),
                "model": T5ForConditionalGeneration.from_pretrained("t5-small", output_hidden_states=True).to(self.device),
            }
        }

        # Small tokenizer fix for GPT-2 (padding)
        self.models["GPT-2"]["tokenizer"].pad_token = self.models["GPT-2"]["tokenizer"].eos_token

    def _filter_predictions(self, raw_predictions, raw_probs):
        """
        Clean a list of raw decoded tokens and probabilities, return cleaned lists.
        raw_predictions: list of strings (decoded tokens)
        raw_probs: list of floats (probabilities)
        """
        cleaned = []
        for tok, p in zip(raw_predictions, raw_probs):
            tok_c = _clean_token_static(tok)
            if _is_reasonable_word(tok_c):
                cleaned.append((tok_c, float(p)))
            # keep going until enough tokens collected
            if len(cleaned) >= self.top_k:
                break

        # fallback: if nothing considered reasonable, return first top_k cleaned tokens (may be short)
        if not cleaned:
            cleaned = []
            for tok, p in zip(raw_predictions, raw_probs):
                tok_c = _clean_token_static(tok)
                # accept anything non-empty
                if tok_c == "":
                    continue
                cleaned.append((tok_c, float(p)))
                if len(cleaned) >= self.top_k:
                    break

        preds, probs = zip(*cleaned) if cleaned else ([], [])
        return list(preds), list(probs)

    def _to_device(self, batch):
        """Move tokenizer batch (dict of tensors) to device"""
        return {k: v.to(self.device) for k, v in batch.items()}

    def evaluate(self, analogy_text: str):
        """
        Runs each model on the analogy_text and returns a dictionary with:
        preds: list of str
        confs: list of float (probabilities)
        hidden: hidden states (or None)
        inputs: the tokenized inputs (on cpu) for visualization use
        """
        results = {}
        for name, comp in self.models.items():
            tokenizer = comp["tokenizer"]
            model = comp["model"]

            if name in ["BERT", "RoBERTa", "DistilBERT"]:
                # Use [MASK] replacement logic
                masked = analogy_text.replace("[MASK]", comp["mask_token"])
                batch = tokenizer(masked, return_tensors="pt")
                batch_dev = self._to_device(batch)
                with torch.no_grad():
                    outputs = model(**batch_dev)
                logits = outputs.logits  # shape (batch, seq_len, vocab)
                mask_token_id = tokenizer.mask_token_id
                mask_positions = (batch_dev["input_ids"] == mask_token_id).nonzero(as_tuple=True)
                try:
                    mask_index = int(mask_positions[1][0].cpu().item())
                except Exception:
                    mask_index = logits.shape[1] - 1

                logits_pos = logits[0, mask_index, :].detach().cpu()
                probs = softmax(logits_pos, dim=-1)
                topk = torch.topk(probs, self.top_k)
                raw_preds = [tokenizer.decode([int(i)]).strip() for i in topk.indices]
                raw_probs = [float(v) for v in topk.values]
                preds, confs = self._filter_predictions(raw_preds, raw_probs)

                hidden = [h.detach().cpu() for h in outputs.hidden_states] if outputs.hidden_states is not None else None
                results[name] = {
                    "preds": preds,
                    "confs": confs,
                    "hidden": hidden,
                    "inputs": batch  # keep CPU version
                }
            elif name == "GPT-2":
                model.eval()
                prompt = analogy_text

                # Remove the mask fully:
                prompt = prompt.replace("[MASK]", "")

                # Convert analogy format to GPT-2 friendly cause-and-effect text:
                prompt = prompt.replace("just like", ". Similarly,")
                prompt = prompt[:-1]

                # Make sure there is a space after "a" or "an"
                if not prompt.endswith(" "):
                    prompt += " "                
                
                print(prompt)
                gpt2_inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    gpt2_outputs = model.generate(
                        **gpt2_inputs,
                        max_length=gpt2_inputs["input_ids"].shape[1] + 3,
                        num_return_sequences=10,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id
                    )
                gpt2_preds = []
                for out in gpt2_outputs:
                    text_out = tokenizer.decode(out[gpt2_inputs["input_ids"].shape[1]:]).strip()
                    first_phrase = " ".join(text_out.split()[:1])
                    first_phrase = first_phrase.replace(tokenizer.eos_token, "").strip()
                    gpt2_preds.append(first_phrase)

                preds, confs = self._filter_predictions(gpt2_preds, raw_probs)

                results[name] = {
                    "preds": preds,
                    "confs": confs,
                    "hidden": None,  # GPT-2 hidden states not used here
                    "inputs": gpt2_inputs  # store tokenized prompt
                }


            else:  # T5 (encoder-decoder)
                prompt = f"Fill in the blank: {analogy_text.replace('[MASK]', '___')}"
                batch = tokenizer(prompt, return_tensors="pt")
                batch_dev = self._to_device(batch)
                with torch.no_grad():
                    gen = model.generate(**batch_dev, max_new_tokens=2)
                decoded = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
                decoded_clean = _clean_token_static(decoded)
                candidate = decoded_clean.split()[-1] if " " in decoded_clean else decoded_clean
                if not _is_reasonable_word(candidate):
                    parts = re.findall(r"[A-Za-z'-]+", decoded)
                    candidate = parts[-1] if parts else decoded_clean
                results[name] = {
                    "preds": [candidate],
                    "confs": [1.0],
                    "hidden": None,
                    "inputs": batch
                }

        return results


# -------------------------
# Visualization helpers
# -------------------------
def plot_confidences(preds, confs, model):
    if not preds:
        st.info(f"No predictions to display for {model}.")
        return
    fig = px.bar(x=preds, y=confs, title=f"Prediction Confidences ({model})",
                 labels={'x': 'Prediction', 'y': 'Confidence'})
    st.plotly_chart(fig, use_container_width=True)

def plot_embeddings(hidden_states, tokenizer, input_ids, model):
    if hidden_states is None:
        st.info(f"{model} does not support embedding visualization.")
        return
    tokens = tokenizer.convert_ids_to_tokens(input_ids["input_ids"][0])
    embeddings = hidden_states[-1][0].numpy()  # (seq_len, hidden_dim)

    if len(tokens) < 3:
        st.warning("Not enough tokens for t-SNE; falling back to PCA.")
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, len(tokens)-1)))

    reduced = reducer.fit_transform(embeddings)
    fig = px.scatter(x=reduced[:, 0], y=reduced[:, 1], text=tokens,
                     title=f"Token Embeddings ({model})")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

def plot_vector_offset(hidden_states, tokenizer, input_ids, model):
    if hidden_states is None:
        return
    tokens = tokenizer.convert_ids_to_tokens(input_ids["input_ids"][0])
    embeddings = hidden_states[-1][0].numpy()  # (seq_len, hidden_dim)
    if embeddings.shape[0] < 4:
        return
    # Use dimensionality reduction on the first 4 token embeddings
    sample_emb = embeddings[:4]
    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, sample_emb.shape[0]-1)))
    reduced = reducer.fit_transform(sample_emb)
    fig, ax = plt.subplots()
    ax.quiver(reduced[0,0], reduced[0,1], reduced[1,0]-reduced[0,0], reduced[1,1]-reduced[0,1],
              color='r', angles='xy', scale_units='xy', scale=1, width=0.005)
    ax.quiver(reduced[2,0], reduced[2,1], reduced[3,0]-reduced[2,0], reduced[3,1]-reduced[2,1],
              color='g', angles='xy', scale_units='xy', scale=1, width=0.005)
    for i, t in enumerate(tokens[:4]):
        ax.text(reduced[i,0], reduced[i,1], t)
    ax.set_title(f"Analogy Vector Offsets ({model})")
    st.pyplot(fig)

# -------------------------
# Streamlit app layout
# -------------------------
st.set_page_config(page_title="Analogy Reasoning Dashboard", layout="wide")
st.sidebar.title("⚙️ Model Evaluation Settings")

analogy_text = st.sidebar.text_input("Enter analogy prompt (use [MASK]):", "A pilot works for an airline, just like a sailor works for a [MASK].")
top_k = st.sidebar.slider("Top K Predictions", 3, 15, 5)
models_selected = st.sidebar.multiselect("Select Models to Compare:",
                                         ["BERT", "RoBERTa", "DistilBERT", "GPT-2", "T5"],
                                         default=["BERT", "GPT-2"])
run_button = st.sidebar.button("🚀 Run Evaluation")

st.title("🔍 Analogy Reasoning Dashboard")

# store and reuse evaluator in session_state to avoid reloading heavy models
if "evaluator" not in st.session_state:
    st.session_state["evaluator"] = None

if run_button:
    st.session_state["evaluator"] = AnalogyEvaluator(top_k=top_k)
    evaluator = st.session_state["evaluator"]
    with st.spinner("Running model evaluations... (this may take a moment)"):
        results = evaluator.evaluate(analogy_text)
    st.session_state["results"] = results
    st.success("✅ Evaluation complete!")

# show results if available
if "results" in st.session_state:
    results = st.session_state["results"]
    evaluator = st.session_state["evaluator"]

    selected_model = st.selectbox("Select Model to View:", models_selected)

    if selected_model in results:
        preds, confs = results[selected_model]["preds"], results[selected_model]["confs"]
        st.subheader(f"🧩 {selected_model}")
        st.write("**Top Predictions:**", preds)
        st.write("**Confidences:**", confs)

        col1, col2 = st.columns(2)
        with col1:
            plot_confidences(preds, confs, selected_model)
        with col2:
            plot_embeddings(results[selected_model]["hidden"],
                            evaluator.models[selected_model]["tokenizer"],
                            results[selected_model]["inputs"],
                            selected_model)

        plot_vector_offset(results[selected_model]["hidden"],
                           evaluator.models[selected_model]["tokenizer"],
                           results[selected_model]["inputs"],
                           selected_model)
else:
    st.info("👈 Enter an analogy and click **Run Evaluation** to begin.")
