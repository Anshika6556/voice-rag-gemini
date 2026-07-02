"""
GEMINI Voice RAG Agent
Uses: Google Gemini 2.5 Flash (LLM) + gTTS (TTS) + FastEmbed (embeddings) + Qdrant (vector DB)

"""

from typing import List, Tuple, Optional
import os
import tempfile
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Qdrant vector database
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# LangChain for PDF loading and text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# FastEmbed -- runs locally, no API key needed
from fastembed import TextEmbedding

# Google Gemini -- replaces OpenAI GPT-4o
import google.generativeai as genai

# gTTS -- free text-to-speech
from gtts import gTTS
import pygame

# ─── Load environment variables from .env file ─────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "voice-rag-agent"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ─── Session State Initialization ──────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processed_documents": [],
        "tts_lang": "en",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─── Sidebar Configuration ─────────────────────────────────────────────────
def setup_sidebar() -> None:
    with st.sidebar:
        st.title("⚙️ Configuration")

        st.subheader("API Keys")
        gemini_key = st.text_input(
            "Gemini API Key",
            value=GEMINI_API_KEY or "",
            type="password",
            help="Get free key at aistudio.google.com",
        )
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=QDRANT_URL or "",
            type="password",
            help="From your Qdrant Cloud cluster",
        )
        qdrant_key = st.text_input(
            "Qdrant API Key",
            value=QDRANT_API_KEY or "",
            type="password",
        )

        st.subheader("Voice Settings")
        tts_lang = st.selectbox(
            "TTS Language",
            options=["en", "hi", "bn", "es", "fr", "de", "ja", "ko"],
            format_func=lambda x: {
                "en": "English",
                "hi": "Hindi",
                "bn": "Bengali",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "ja": "Japanese",
                "ko": "Korean",
            }.get(x, x),
            index=0,
        )
        st.session_state.tts_lang = tts_lang

        if st.button("🚀 Initialize System", use_container_width=True):
            if not gemini_key or not qdrant_url or not qdrant_key:
                st.error("Please fill in all API keys first.")
                return

            with st.spinner("Connecting to services..."):
                try:
                    genai.configure(api_key=gemini_key)
                    client, embedding_model = setup_qdrant(qdrant_url, qdrant_key)
                    st.session_state.client = client
                    st.session_state.embedding_model = embedding_model
                    st.session_state.setup_complete = True
                    st.success("✅ System ready!")
                except Exception as e:
                    st.error(f"Setup failed: {str(e)}")

        if st.session_state.processed_documents:
            st.subheader("📚 Uploaded Documents")
            for doc_name in st.session_state.processed_documents:
                st.write(f"• {doc_name}")


# ─── Qdrant Vector Database Setup ──────────────────────────────────────────
def setup_qdrant(qdrant_url: str, qdrant_key: str) -> Tuple[QdrantClient, TextEmbedding]:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # Use a permanent cache folder instead of the default Temp folder.
    # Windows/antivirus sometimes clears Temp mid-download, which causes
    # "File doesn't exist" errors on the .onnx model file.
    cache_dir = os.path.join(os.path.expanduser("~"), "fastembed_models")
    os.makedirs(cache_dir, exist_ok=True)
    embedding_model = TextEmbedding(cache_dir=cache_dir)

    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)

    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        st.info(f"Created new collection: {COLLECTION_NAME}")
    else:
        st.info(f"Using existing collection: {COLLECTION_NAME}")

    return client, embedding_model


# ─── PDF Processing ────────────────────────────────────────────────────────
def process_pdf(uploaded_file) -> List:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata.update(
            {
                "source_type": "pdf",
                "file_name": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
            }
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    os.unlink(tmp_path)
    return chunks


def store_embeddings(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    documents: List,
) -> None:
    for doc in documents:
        embedding = list(embedding_model.embed([doc.page_content]))[0]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": doc.page_content,
                        **doc.metadata,
                    },
                )
            ],
        )


# ─── Query Processing with Gemini ──────────────────────────────────────────
def answer_query(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
) -> str:
    query_embedding = list(embedding_model.embed([query]))[0]

    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=3,
        with_payload=True,
    )
    search_results = search_response.points

    if not search_results:
        return "I couldn't find any relevant information in the uploaded documents."

    context = "Here is the relevant documentation context:\n\n"
    for i, result in enumerate(search_results, 1):
        payload = result.payload
        content = payload.get("content", "")
        source = payload.get("file_name", "Unknown document")
        context += f"[Source {i}: {source}]\n{content}\n\n"

    prompt = f"""{context}
---
Based ONLY on the documentation provided above, please answer this question:

Question: {query}

Instructions:
- Answer clearly and conversationally, as if speaking aloud
- Keep the answer concise (2-4 sentences for simple questions)
- If the answer isn't in the documents, say so honestly
- Cite which source you're drawing from when helpful
- Use natural language, avoid bullet points
"""

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=500,
        ),
    )
    response = model.generate_content(prompt)
    return response.text


# ─── Text-to-Speech with gTTS ──────────────────────────────────────────────
def text_to_speech(text: str, lang: str = "en") -> Optional[str]:
    """
    Convert text to speech using gTTS. Completely free, no API key.
    Saves audio to a temp file and plays it via pygame (if a sound
    device is available). 
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tmp_path = os.path.join(tempfile.gettempdir(), f"answer_{uuid.uuid4()}.mp3")
        tts.save(tmp_path)

        try:
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception:
            # No local sound device (e.g. on a server) -- that's fine,
            # the browser audio player below will still work.
            pass

        return tmp_path
    except Exception as e:
        st.warning(f"Audio generation failed: {str(e)}. The text answer is still shown above.")
        return None


# ─── Main Streamlit App ────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Voice RAG Agent",
        page_icon="🎙",
        layout="wide",
    )

    init_session_state()
    setup_sidebar()

    st.title("🎙 Voice RAG Agent")
    st.caption("Powered by Google Gemini + gTTS")

    if not st.session_state.setup_complete:
        st.info("👈 Enter your API keys in the sidebar and click **Initialize System** to begin.")
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📄 Upload a Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload any PDF document you want to ask questions about",
        )

        if uploaded_file is not None:
            file_already_processed = uploaded_file.name in st.session_state.processed_documents
            if file_already_processed:
                st.success(f"✅ {uploaded_file.name} is already processed.")
            else:
                if st.button("⚙️ Process Document", use_container_width=True):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            chunks = process_pdf(uploaded_file)

                            progress_bar = st.progress(0)
                            for i, chunk in enumerate(chunks):
                                store_embeddings(
                                    st.session_state.client,
                                    st.session_state.embedding_model,
                                    [chunk],
                                )
                                progress_bar.progress((i + 1) / len(chunks))

                            st.session_state.processed_documents.append(uploaded_file.name)
                            st.success(f"✅ Stored {len(chunks)} chunks from {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")

    with col2:
        st.subheader("🎤 Ask a Question")

        if not st.session_state.processed_documents:
            st.warning("Please upload and process a document first.")
        else:
            query = st.text_area(
                "Your question:",
                placeholder="e.g., What is the main topic of this document?",
                height=100,
                help="Ask anything about your uploaded documents",
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                speak_btn = st.button(
                    "🔊 Get Voice Answer",
                    use_container_width=True,
                    type="primary",
                )
            with col_btn2:
                text_btn = st.button("📝 Text Only", use_container_width=True)

            if (speak_btn or text_btn) and query.strip():
                with st.spinner("🤔 Searching documents and generating answer..."):
                    try:
                        answer = answer_query(
                            query,
                            st.session_state.client,
                            st.session_state.embedding_model,
                        )

                        st.subheader("💬 Answer")
                        st.write(answer)

                        if speak_btn:
                            with st.spinner("🔊 Generating audio..."):
                                audio_path = text_to_speech(answer, lang=st.session_state.tts_lang)
                                if audio_path and os.path.exists(audio_path):
                                    with open(audio_path, "rb") as audio_file:
                                        audio_bytes = audio_file.read()
                                    st.audio(audio_bytes, format="audio/mp3")
                                    st.download_button(
                                        label="⬇️ Download MP3",
                                        data=audio_bytes,
                                        file_name="answer.mp3",
                                        mime="audio/mp3",
                                    )
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                        st.info("Check that your Gemini API key is valid and has quota remaining.")
            elif (speak_btn or text_btn) and not query.strip():
                st.warning("Please enter a question first.")

    st.divider()
    st.caption("Built with Gemini 2.5 Flash + gTTS + FastEmbed + Qdrant - Anshika Pandey")



if __name__ == "__main__":
    main()