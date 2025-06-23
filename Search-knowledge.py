# app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.preprocessing import normalize
import openai

# --- CONFIG ---
st.set_page_config(page_title="RAG Kenniszoeker", layout="wide")

# --- TITLE ---
st.title("🔍 RAG Kenniszoeker met Groq")

# --- SIDEBAR: API KEY ---
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload je Knowledge CSV", type=["csv"])

if uploaded_file and groq_api_key:
    # Init Groq
    openai.api_key = groq_api_key

    # Lees CSV
    df = pd.read_csv(uploaded_file)

    # Voeg embedding input samen
    df['embedding_input'] = df.apply(
        lambda row: f"{row['title']} {row['summary']} {row['UrlName']}",
        axis=1
    )
    texts = df['embedding_input'].tolist()

    # Laad sentence transformer
    st.write("🔗 Embeddings genereren...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = normalize(embeddings.astype("float32"))

    # FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.success("✅ Index klaar! Stel je vraag.")

    # --- Vraag ---
    vraag = st.text_input("Stel je vraag")

    if st.button("Zoek en beantwoord"):
        if vraag.strip() == "":
            st.warning("Typ eerst een vraag!")
        else:
            # Embed vraag
            q_emb = model.encode([vraag], convert_to_numpy=True)
            q_emb = normalize(q_emb.astype("float32"))

            # Zoek top 3
            D, I = index.search(q_emb, k=3)
            matches = df.iloc[I[0]]
            context = "\n\n".join(matches['embedding_input'])

            # Prompt
            prompt = f"Beantwoord de vraag op basis van deze artikelen:\n\n{context}\n\nVraag: {vraag}\nAntwoord:"

            # Groq LLM call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            antwoord = response['choices'][0]['message']['content']

            st.subheader("🔗 Relevante artikelen")
            st.dataframe(matches[['Title', 'UrlName']].reset_index(drop=True))
            st.subheader("💡 Antwoord")
            st.write(antwoord)
else:
    st.info("➡️ Upload een CSV en vul je Groq API key in om te starten.")
