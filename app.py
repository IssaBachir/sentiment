import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import time

st.set_page_config(page_title="Analyse de Sentiment", page_icon="💬", layout="centered")

# Initialisation de la session
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_model():
    model_path = "./models/distilbert-sentiment"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label].item()
    return pred_label, confidence

# UI principale
st.title("💬 Analyse de Sentiment")
st.markdown("**Modèle utilisé** : DistilBERT — binaire (positif / négatif)")

text = st.text_area("📝 Entrez un texte ici :", height=150)

if st.button("🔍 Analyser le sentiment"):
    if not text.strip():
        st.warning("🚨 Merci d’écrire quelque chose.")
    else:
        with st.spinner("Analyse en cours..."):
            tokenizer, model = load_model()
            label, conf = predict(text, tokenizer, model)
            sentiment = "🟢 Positif" if label == 1 else "🔴 Négatif"

            st.success(f"**Résultat : {sentiment}**")
            st.progress(conf)
            st.write(f"Confiance : **{conf*100:.2f}%**")

            # Historique
            st.session_state.history.append((text, sentiment, conf))

# Historique de session
if st.session_state.history:
    st.subheader("📜 Historique de session")
    for i, (txt, sent, conf) in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**Texte {len(st.session_state.history)-i}** : _{txt}_")
        st.markdown(f"➡️ Sentiment : {sent} — Confiance : `{conf*100:.2f}%`")
        st.markdown("---")
