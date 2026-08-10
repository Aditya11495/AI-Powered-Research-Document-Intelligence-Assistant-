import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_engine import RAGEngine

st.set_page_config(
    page_title="AI Business & Research RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Business & Research RAG Assistant")
st.caption("Multi-document RAG with semantic retrieval, citations, and retrieval-grounded answers")

@st.cache_resource(show_spinner=False)
def get_engine():
    return RAGEngine()

engine = get_engine()

with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Gemini", "Local / Demo"])
    if provider == "OpenAI":
        model = st.text_input("Model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    elif provider == "Gemini":
        model = st.text_input("Model", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    else:
        model = "demo"

    top_k = st.slider("Retrieved chunks", 2, 10, 5)
    st.divider()
    st.markdown("**Recommended documents**")
    st.caption("Annual reports • research papers • business reports • technical documentation")

uploaded = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDFs to build a searchable knowledge base.",
)

if uploaded:
    with st.spinner("Processing documents and building the vector index..."):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for file in uploaded:
                p = Path(tmp) / file.name
                p.write_bytes(file.getbuffer())
                paths.append(str(p))
            stats = engine.ingest(paths)

    st.success(
        f"Indexed {stats['documents']} document(s), "
        f"{stats['pages']} page(s), and {stats['chunks']} chunks."
    )

st.subheader("Ask your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for s in message["sources"]:
                    st.markdown(
                        f"- **{s['source']}**, page **{s['page']}** "
                        f"(similarity: `{s['score']:.3f}`)"
                    )

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if not engine.has_index():
        st.warning("Please upload at least one PDF before asking a question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating an answer..."):
            result = engine.answer(
                question=question,
                provider=provider,
                model=model,
                top_k=top_k,
            )
        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📎 Evidence & citations", expanded=True):
                for s in result["sources"]:
                    st.markdown(
                        f"**{s['source']} — page {s['page']}**  \n"
                        f"Similarity: `{s['score']:.3f}`  \n"
                        f"> {s['snippet']}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
