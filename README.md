# 🎙 Voice RAG Agent 

A Voice-powered Retrieval-Augmented Generation (RAG) app. Upload a PDF, ask
questions about it in plain text, and get answers spoken aloud.

## What It Does

- 📄 Upload any PDF document
- 🔍 The document is split into chunks and stored as vectors in a Qdrant
  vector database
- 💬 Ask questions about the document in natural language
- 🧠 Google Gemini 2.5 Flash finds the relevant chunks and generates an
  answer
- 🔊 The answer is read aloud using free text-to-speech (gTTS)


## Tech Stack

| Component     | Tool                                  |
|----------------|----------------------------------------|
| LLM            | Google Gemini 2.5 Flash                |
| Embeddings     | FastEmbed (runs locally, no API key)   |
| Vector DB      | Qdrant Cloud (free tier)               |
| Text-to-Speech | gTTS (Google Text-to-Speech)           |
| UI             | Streamlit                              |
| PDF Parsing    | LangChain + PyPDFLoader                |

## 🔑 Bring Your Own API Keys

This app does **not** ship with any API keys built in. Each user enters
their own keys directly in the app's sidebar when they open it:

1. A free **Gemini API key** — get one at https://aistudio.google.com
2. A free **Qdrant Cloud URL + API key** — get one at https://cloud.qdrant.io

Your keys are only used in your own browser session and are never stored
or sent anywhere except directly to Google/Qdrant's official APIs.

## Running Locally

### 1. Clone this repository

```bash
git clone https://github.com/YOUR-USERNAME/voice-rag-gemini.git
cd voice-rag-gemini
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Create a `.env` file for local convenience

If you don't want to type your keys into the sidebar every time locally,
create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_key_here
QDRANT_URL=your_qdrant_cluster_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

This file is git-ignored and never gets pushed to GitHub.

### 5. Run the app

```bash
streamlit run voice_rag_agent.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

### 6. Use the app

1. Enter your Gemini and Qdrant keys in the sidebar (auto-filled if you
   set up `.env`)
2. Click **🚀 Initialize System**
3. Upload a PDF and click **⚙️ Process Document**
4. Type a question and click **🔊 Get Voice Answer** or **📝 Text Only**


Free to use and modify for personal or educational projects.
