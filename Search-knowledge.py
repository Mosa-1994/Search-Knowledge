import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from groq import Groq
import re
import os

# --- CONFIG ---
st.set_page_config(page_title="RAG Kenniszoeker", layout="wide")

# --- TITLE ---
st.title("🔍 RAG Kenniszoeker met Groq")

# --- SIDEBAR ---
index_option = st.sidebar.radio(
    "Index-optie",
    ("📄 Nieuw: upload CSV", "📁 Bestaand: laad opgeslagen index")
)

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# --- Functie: opschonen ---
def clean_text(text):
    text = str(text)
    text = re.sub('<.*?>', '', text)  # HTML weg
    text = re.sub(r'\s+', ' ', text)  # dubbele spaties
    return text.strip()

# --- Simpele Nederlands-vriendelijke zinsplitser ---
def sent_tokenize(text):
    # Split op punt, vraagteken of uitroepteken + spatie/einde
    return re.split(r'(?<=[.!?])\s+', text)

# --- Functie: splitsen in chunks ---
def chunk_article(row):
    title = clean_text(row['Title'])
    summary = clean_text(row['Summary'])
    body = clean_text(row['ArticleBody'])
    urlname = row['UrlName']

    sentences = sent_tokenize(body)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) < 100:
            chunk += " " + sentence
        else:
            chunks.append({
                'Title': title,
                'Summary': summary,
                'UrlName': urlname,
                'embedding_input': f"{title}. {summary}. {chunk.strip()}"
            })
            chunk = sentence
    if chunk.strip():
        chunks.append({
            'Title': title,
            'Summary': summary,
            'UrlName': urlname,
            'embedding_input': f"{title}. {summary}. {chunk.strip()}"
        })
    return chunks

# --- ACTIE: Laad bestaande index ---
if index_option == "📁 Bestaand: laad opgeslagen index" and groq_api_key:
    try:
        df = pd.read_csv("index_data/chunks.csv")
        embeddings = np.load("index_data/embeddings.npy")
        index = faiss.read_index("index_data/faiss.index")
        st.success("✅ Bestaande index geladen.")
    except Exception as e:
        st.error(f"❌ Fout bij laden: {e}")
        st.stop()

# --- ACTIE: Upload nieuwe CSV & maak index ---
elif index_option == "📄 Nieuw: upload CSV" and groq_api_key:
    uploaded_file = st.file_uploader("Upload je Knowledge CSV", type=["csv"])
    if uploaded_file:
        # Init Groq Client
        client = Groq(api_key=groq_api_key)

        # Lees CSV en chunk
        df_raw = pd.read_csv(uploaded_file, sep=';')
        all_chunks = []
        for idx, row in df_raw.iterrows():
            all_chunks.extend(chunk_article(row))
        df = pd.DataFrame(all_chunks)

        texts = df['embedding_input'].tolist()

        # Embeddings genereren
        st.write("🔗 Embeddings genereren...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = normalize(embeddings.astype("float32"))

        # FAISS index bouwen
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Opslaan
        os.makedirs("index_data", exist_ok=True)
        np.save("index_data/embeddings.npy", embeddings)
        df.to_csv("index_data/chunks.csv", index=False)
        faiss.write_index(index, "index_data/faiss.index")
        st.success("✅ Nieuwe index gemaakt en opgeslagen in 'index_data/'.")

    else:
        st.info("➡️ Upload een CSV en vul je API key in.")
        st.stop()

# --- Beëindig als er geen geldige combinatie is ---
else:
    st.info("➡️ Kies een optie en vul je Groq API key in.")
    st.stop()

# --- Client init als nog niet ---
client = Groq(api_key=groq_api_key)

# --- Vraag ---
vraag = st.text_input("Stel je vraag")

if st.button("Zoek en beantwoord"):
    if vraag.strip() == "":
        st.warning("Typ eerst een vraag!")
    else:
        # Embed vraag
        model = SentenceTransformer('all-MiniLM-L6-v2')
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
                model="llama3-8b-8192",
            )
            antwoord = chat_completion.choices[0].message.content
        except Exception as e:
            st.error(f"Fout bij het aanroepen van de Groq API: {e}")
            antwoord = "Er is een fout opgetreden bij het genereren van het antwoord."

        st.subheader("🔗 Relevante artikelen")
        st.dataframe(matches[['Title', 'Summary', 'UrlName']].reset_index(drop=True))
        st.subheader("💡 Antwoord")
        st.write(antwoord)
