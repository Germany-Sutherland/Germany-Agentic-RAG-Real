# agent_pipeline.py
"""
Multi-agent Agentic RAG pipeline:
- PlannerAgent (generates short plan steps with small generator)
- RetrieverAgent (sentence-transformers embeddings + FAISS optional, NumPy fallback)
- QAAgent (DistilBERT QA executor)
- VerifierAgent (simple heuristic verification)
- SynthesizerAgent (paraphrase/combine results)
- SentimentAgent (question and answer sentiment)
Designed with robust fallbacks and suppressed logging to avoid noisy warnings.
"""

import logging, warnings
from typing import List, Dict

# suppress noisy warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Lazy loaders to reduce cold import footprint on deploy
def _load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def _try_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None

def _load_qa_pipeline():
    from transformers import pipeline
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def _load_generator():
    from transformers import pipeline, set_seed
    gen = pipeline("text-generation", model="distilgpt2")
    set_seed(42)
    return gen

def _load_sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis")

# ----------------- Retriever Agent -----------------
class RetrieverAgent:
    def __init__(self, documents: List[str]):
        self.docs = list(documents)
        self.embedder = _load_embedder()
        self.faiss = _try_faiss()
        self._build_index()

    def _build_index(self):
        import numpy as np
        embs = self.embedder.encode(self.docs, convert_to_numpy=True)
        if self.faiss:
            dim = embs.shape[1]
            self.index = self.faiss.IndexFlatL2(dim)
            self.index.add(embs.astype("float32"))
            self.mode = "faiss"
        else:
            self.embs = embs
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
            self.norms = embs / norms
            self.mode = "numpy"

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        import numpy as np
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        if self.mode == "faiss":
            D, I = self.index.search(q_emb.astype("float32"), k)
            return [self.docs[i] for i in I[0] if i < len(self.docs)]
        else:
            qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
            sims = (self.norms @ qn.T).squeeze()
            idxs = sims.argsort()[::-1][:k]
            return [self.docs[int(i)] for i in idxs]

# ----------------- Planner Agent -----------------
class PlannerAgent:
    def __init__(self):
        self.gen = _load_generator()

    def plan(self, query: str) -> List[str]:
        prompt = f"User asks: {query}\nWrite 2 concise numbered steps the agent should take."
        try:
            out = self.gen(prompt, max_length=64, num_return_sequences=1)[0]["generated_text"]
            steps_text = out.replace(prompt, "").strip()
            lines = [l.strip("-. 0123456789") for l in steps_text.splitlines() if l.strip()]
            return lines[:3] if lines else ["Retrieve relevant docs", "Answer from context"]
        except Exception:
            return ["Retrieve relevant docs", "Answer from context"]

# ----------------- QA Agent -----------------
class QAAgent:
    def __init__(self):
        self.qa = _load_qa_pipeline()

    def answer(self, question: str, context: str) -> Dict:
        try:
            out = self.qa({"question": question, "context": context})
            return {"answer": out.get("answer", ""), "score": out.get("score", 0.0)}
        except Exception as e:
            return {"answer": "", "score": 0.0, "error": str(e)}

# ----------------- Verifier Agent -----------------
class VerifierAgent:
    def verify(self, candidate: str, retrieved: List[str]) -> Dict:
        ok = bool(candidate and any(candidate.strip()[:6].lower() in r.lower() for r in retrieved))
        return {"ok": ok, "note": "heuristic: presence check"}

# ----------------- Synthesizer Agent -----------------
class SynthesizerAgent:
    def __init__(self):
        self.gen = _load_generator()

    def synthesize(self, question: str, pieces: List[str]) -> str:
        context = " ||| ".join([p for p in pieces if p])
        prompt = f"Question: {question}\nFacts: {context}\nWrite a concise 1-2 sentence answer:"
        try:
            out = self.gen(prompt, max_length=120, num_return_sequences=1)[0]["generated_text"]
            ans = out.replace(prompt, "").strip()
            return ans if ans else " ".join(pieces[:2])
        except Exception:
            return " ".join(pieces[:2])

# ----------------- Sentiment Agent -----------------
class SentimentAgent:
    def __init__(self):
        try:
            self.pipe = _load_sentiment()
        except Exception:
            self.pipe = None

    def get_sentiment(self, text: str) -> Dict:
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}
        try:
            if not self.pipe:
                return {"label": "UNKNOWN", "score": 0.0}
            out = self.pipe(text[:512])[0]
            return {"label": out.get("label", "UNKNOWN"), "score": float(out.get("score", 0.0))}
        except Exception:
            return {"label": "UNKNOWN", "score": 0.0}

# ----------------- Orchestrator AgenticRAG -----------------
class AgenticRAG:
    def __init__(self, documents: List[str]):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(documents)
        self.executor = QAAgent()
        self.verifier = VerifierAgent()
        self.synth = SynthesizerAgent()
        self.sentiment = SentimentAgent()

    def run(self, question: str, top_k: int = 2) -> Dict:
        plan = self.planner.plan(question)
        retrieved = self.retriever.retrieve(question, k=top_k)
        context = "\n\n".join(retrieved)
        qa_out = self.executor.answer(question, context)
        verification = self.verifier.verify(qa_out.get("answer", ""), retrieved)
        final = self.synth.synthesize(question, [qa_out.get("answer", "")] + retrieved)
        q_sent = self.sentiment.get_sentiment(question)
        a_sent = self.sentiment.get_sentiment(final)
        return {
            "plan": plan,
            "retrieved": retrieved,
            "qa": qa_out,
            "verification": verification,
            "answer": final,
            "question_sentiment": q_sent,
            "answer_sentiment": a_sent
        }
