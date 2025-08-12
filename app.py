# app.py
import streamlit as st
from textwrap import shorten
from job_data import DOCUMENTS

st.set_page_config(page_title="Agentic RAG — Germany (Multi-agent Demo)", layout="wide")
st.title("🇩🇪 Agentic RAG — Germany Job Market (Multi-agent Demo + Sentiment & History)")

@st.cache_resource(show_spinner=False)
def init_agent():
    from agent_pipeline import AgenticRAG
    return AgenticRAG(DOCUMENTS)

agent = init_agent()

# history (last 5 questions)
if "history" not in st.session_state:
    st.session_state["history"] = []

st.markdown("Ask about jobs, skills, industries or cities in Germany. The agent will plan, retrieve, execute QA, verify and synthesize an answer. Sentiment detection and last-5 history are included.")

col_main, col_ctrl = st.columns([3,1])
with col_ctrl:
    top_k = st.selectbox("Docs to retrieve (k)", [1,2,3], index=1)
    show_plan = st.checkbox("Show plan", True)
    show_retrieved = st.checkbox("Show retrieved", True)
    show_qa = st.checkbox("Show QA output", False)
    show_verification = st.checkbox("Show verification", True)
    show_sentiment = st.checkbox("Show sentiment", True)

with col_main:
    question = st.text_input("Type your question here:")
    if st.button("Run Agent") and question.strip():
        with st.spinner("Agent is planning and executing..."):
            out = agent.run(question, top_k)
        # record history
        st.session_state["history"].append(question)
        st.session_state["history"] = st.session_state["history"][-5:]

        # show plan
        if show_plan:
            st.subheader("Plan (agent steps)")
            for i, step in enumerate(out["plan"], 1):
                st.write(f"{i}. {step}")

        # final answer
        st.subheader("Final answer")
        st.success(shorten(out["answer"], width=700))

        # sentiment
        if show_sentiment:
            qs = out.get("question_sentiment", {})
            anss = out.get("answer_sentiment", {})
            st.markdown("**Sentiment**")
            st.write(f"Question sentiment: **{qs.get('label','UNKNOWN')}** (score {qs.get('score',0.0):.2f})")
            st.write(f"Answer sentiment: **{anss.get('label','UNKNOWN')}** (score {anss.get('score',0.0):.2f})")

        # QA details
        if show_qa:
            st.markdown("**QA output**")
            st.json(out["qa"])

        # verification
        if show_verification:
            st.markdown("**Verification**")
            st.json(out["verification"])

        # retrieved
        if show_retrieved:
            st.markdown("**Retrieved passages**")
            for i, p in enumerate(out["retrieved"], 1):
                st.write(f"Doc {i}: {shorten(p, 400)}")

# show last 5 questions
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("Last 5 questions")
    for h in reversed(st.session_state["history"]):
        st.write(f"- {h}")

st.markdown("---")
st.caption("Multi-agent RAG demo built with sentence-transformers & Hugging Face (free & open-source).")


from evaluation_metrics import evaluate_text

# Compute evaluation metrics
scores = evaluate_text(question, result)

st.subheader("Evaluation Metrics")
for metric, value in scores.items():
    st.write(f"{metric}: {value:.4f}")

