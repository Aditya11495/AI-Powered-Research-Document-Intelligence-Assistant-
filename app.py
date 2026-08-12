import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Business & Research RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📚 AI Business & Research RAG Assistant")

st.caption(
    "Multi-document RAG with semantic retrieval, "
    "citations, and retrieval-grounded answers"
)


# ============================================================
# RAG ENGINE
# ============================================================

@st.cache_resource(show_spinner="Loading AI models...")
def get_engine():
    return RAGEngine()


try:
    engine = get_engine()

except Exception as e:
    st.error("Failed to initialize the RAG engine.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        ["OpenAI", "Gemini", "Local / Demo"],
    )

    if provider == "OpenAI":

        model = st.text_input(
            "Model",
            value=os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini",
            ),
        )

    elif provider == "Gemini":

        model = st.text_input(
            "Model",
            value=os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash",
            ),
        )

    else:

        model = "demo"

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=5,
    )

    st.divider()

    st.markdown("**Recommended documents**")

    st.caption(
        "Annual reports • research papers • "
        "business reports • technical documentation"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help=(
        "Upload one or more PDFs to build "
        "a searchable knowledge base."
    ),
)


# ============================================================
# PROCESS UPLOADED DOCUMENTS
# ============================================================

if uploaded:

    with st.spinner(
        "Processing documents and building the vector index..."
    ):

        try:

            with tempfile.TemporaryDirectory() as tmp:

                paths = []

                for file in uploaded:

                    file_path = (
                        Path(tmp) / file.name
                    )

                    file_path.write_bytes(
                        file.getbuffer()
                    )

                    paths.append(
                        str(file_path)
                    )

                stats = engine.ingest(paths)

            st.success(
                f"Indexed {stats['documents']} "
                f"document(s), "
                f"{stats['pages']} page(s), and "
                f"{stats['chunks']} chunks."
            )

        except Exception as e:

            st.error(
                "Failed to process the uploaded PDF(s)."
            )

            st.exception(e)


# ============================================================
# INDEX STATUS
# ============================================================

if engine.has_index():

    st.success(
        f"Knowledge base ready — "
        f"{len(engine.records)} chunks indexed."
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.subheader("Ask your documents")


if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "📎 Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"- **{source['source']}**, "
                        f"page **{source['page']}** "
                        f"(similarity: "
                        f"`{source['score']:.3f}`)"
                    )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    # --------------------------------------------------------
    # CHECK INDEX
    # --------------------------------------------------------

    if not engine.has_index():

        st.warning(
            "Please upload at least one PDF "
            "before asking a question."
        )

        st.stop()

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving evidence and generating an answer..."
        ):

            try:

                result = engine.answer(
                    question=question,
                    provider=provider,
                    model=model,
                    top_k=top_k,
                )

            except Exception as e:

                st.error(
                    "An error occurred while "
                    "generating the answer."
                )

                st.exception(e)

                st.stop()

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(
            result["answer"]
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if result["sources"]:

            with st.expander(
                "📎 Evidence & Citations",
                expanded=True,
            ):

                for source in result["sources"]:

                    st.markdown(
                        f"**{source['source']} — "
                        f"page {source['page']}**  \n"
                        f"Similarity: "
                        f"`{source['score']:.3f}`  \n"
                        f"> {source['snippet']}"
                    )

    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )