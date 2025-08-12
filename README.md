# Agentic RAG — Germany Job Market (Multi-agent Demo)

This demo implements a multi‑agent Agentic RAG pipeline (Planner → Retriever → Executor → Verifier → Synthesizer) using **only free & open-source** components:
- sentence-transformers (embeddings)
- Hugging Face pipelines (QA, generator, sentiment)
- NumPy / optional FAISS for retrieval fallback
- Streamlit front-end for demo & sharing

How to deploy:
1. Deploy this repo on Streamlit Cloud (share.streamlit.io).
2. Main file: `app.py`. Dependencies: `requirements.txt`.

Notes:
- The app uses small open models (distilgpt2 / distilbert / all-MiniLM-L6-v2) to keep CPU usage manageable on Streamlit free tier.
- If FAISS or torch are available in the runtime, the retriever will use them; otherwise it uses a NumPy fallback.
