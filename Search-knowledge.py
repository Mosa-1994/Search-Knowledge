# app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from groq import Groq # Importeer de Groq SDK

# --- CONFIG ---
st.set_page_config(page_title="RAG Kenniszoeker", layout="wide")

# --- TITLE ---
st.title("🔍 RAG Kenniszoeker met Groq")

# --- SIDEBAR: API KEY ---
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload je Knowledge CSV", type=["csv"])

if uploaded_file and groq_api_key:
    # Init Groq Client
    client = Groq(
        api_key=groq_api_key,
    )

    # Lees CSV
    df = pd.read_csv(uploaded_file, sep=';')

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
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="llama3-8b-8192", # Of een ander model dat Groq aanbiedt, bijv. "mixtral-8x7b-32768"
                )
                antwoord = chat_completion.choices[0].message.content
            except Exception as e:
                st.error(f"Fout bij het aanroepen van de Groq API: {e}")
                antwoord = "Er is een fout opgetreden bij het genereren van het antwoord."

            st.subheader("🔗 Relevante artikelen")
            st.dataframe(matches[['title', 'UrlName']].reset_index(drop=True)) # Let op: 'Title' moet waarschijnlijk 'title' zijn
            st.subheader("💡 Antwoord")
            st.write(antwoord)
else:
    st.info("➡️ Upload een CSV en vul je Groq API key in om te starten.")
