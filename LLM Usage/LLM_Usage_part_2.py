import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Sélection et chargement du modèle en local
MODEL_NAME = "google/flan-t5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

# 2. Fonction de génération locale
def query_local(question: str) -> str:
    prompt = f"Answer the following question: {question}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Vous pouvez ajuster max_new_tokens pour plus de longévité de réponse
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Interface Streamlit
st.set_page_config(page_title="🤖 Local LLM Chatbot", page_icon="🤖")
st.title("🤖 LLM Chatbot en Local (Flan-T5-Small)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me anything!")

if user_input:
    # On stocke l'entrée utilisateur
    st.session_state.history.append(("user", user_input))
    # On génère la réponse en local
    with st.spinner("Génération en cours…"):
        bot_response = query_local(user_input)
    st.session_state.history.append(("bot", bot_response))

# Affichage de l’historique
for sender, message in st.session_state.history:
    if sender == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
