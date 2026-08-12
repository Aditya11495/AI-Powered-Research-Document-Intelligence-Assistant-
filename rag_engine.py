import json
import os
import re
from pathlib import Path

import faiss
import fitz
from sentence_transformers import SentenceTransformer


SYSTEM_PROMPT = """You are a careful business and research document assistant.

Answer ONLY from the supplied evidence.

Rules:
1. Do not invent facts.
2. If the evidence is insufficient, clearly say that the information is not available in the supplied documents.
3. Cite important claims using [Source, p.X].
4. Keep answers concise but useful.
5. When comparing entities, use a table when helpful.
"""


class RAGEngine:
    """Retrieval-Augmented Generation engine using FAISS and Sentence Transformers."""

    def __init__(self):
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.index_dir = Path(
            os.getenv("INDEX_DIR", "data/index")
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index = None
        self.records = []

        # Load embedding model
        self.embedder = SentenceTransformer(
            self.embedding_model_name
        )

        # Load existing FAISS index if available
        self._load()

    # =========================================================
    # TEXT PROCESSING
    # =========================================================

    @staticmethod
    def _clean(text):
        """Clean extracted PDF text."""
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _chunk(text, chunk_size=900, overlap=150):
        """Split text into overlapping word-based chunks."""

        words = text.split()

        if not words:
            return []

        chunks = []
        start = 0

        while start < len(words):
            end = min(
                start + chunk_size,
                len(words),
            )

            chunks.append(
                " ".join(words[start:end])
            )

            if end == len(words):
                break

            start = end - overlap

        return chunks

    # =========================================================
    # PDF INGESTION
    # =========================================================

    def ingest(self, pdf_paths):
        """
        Read PDFs, create chunks, generate embeddings,
        and build a FAISS index.
        """

        records = []

        for pdf_path in pdf_paths:

            pdf_path = Path(pdf_path)

            if not pdf_path.exists():
                continue

            try:
                document = fitz.open(pdf_path)

                for page_number, page in enumerate(
                    document,
                    start=1,
                ):
                    text = self._clean(
                        page.get_text("text")
                    )

                    if not text:
                        continue

                    chunks = self._chunk(text)

                    for chunk_id, chunk in enumerate(chunks):

                        records.append(
                            {
                                "text": chunk,
                                "source": pdf_path.name,
                                "page": page_number,
                                "chunk_id": chunk_id,
                            }
                        )

                document.close()

            except Exception as exc:
                raise RuntimeError(
                    f"Could not process PDF '{pdf_path.name}': {exc}"
                ) from exc

        if not records:
            raise ValueError(
                "No readable text was found in the uploaded PDFs."
            )

        # Generate embeddings
        texts = [
            record["text"]
            for record in records
        ]

        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Create FAISS index
        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(embeddings)

        # Store in memory
        self.index = index
        self.records = records

        # Persist index
        index_path = (
            self.index_dir / "vectors.faiss"
        )

        records_path = (
            self.index_dir / "records.json"
        )

        faiss.write_index(
            index,
            str(index_path),
        )

        records_path.write_text(
            json.dumps(
                records,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "documents": len(
                set(
                    record["source"]
                    for record in records
                )
            ),
            "pages": len(
                set(
                    (
                        record["source"],
                        record["page"],
                    )
                    for record in records
                )
            ),
            "chunks": len(records),
        }

    # =========================================================
    # INDEX LOADING
    # =========================================================

    def _load(self):
        """Load an existing FAISS index and metadata."""

        index_path = (
            self.index_dir / "vectors.faiss"
        )

        records_path = (
            self.index_dir / "records.json"
        )

        if not (
            index_path.exists()
            and records_path.exists()
        ):
            return

        try:
            self.index = faiss.read_index(
                str(index_path)
            )

            self.records = json.loads(
                records_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            # If an old/corrupted index exists,
            # start cleanly without crashing the app.
            self.index = None
            self.records = []

    # =========================================================
    # INDEX STATUS
    # =========================================================

    def has_index(self):
        """Return True when a valid document index exists."""

        return (
            self.index is not None
            and len(self.records) > 0
        )

    def clear_index(self):
        """Clear the current in-memory index."""

        self.index = None
        self.records = []

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def retrieve(self, question, top_k=5):
        """
        Retrieve the most relevant document chunks.
        """

        if not self.has_index():
            raise RuntimeError(
                "No document index is available. "
                "Please upload at least one PDF."
            )

        if not question or not question.strip():
            return []

        top_k = max(
            1,
            min(
                int(top_k),
                len(self.records),
            ),
        )

        # Embed user question
        query_embedding = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Search FAISS
        scores, ids = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index_id in zip(
            scores[0],
            ids[0],
        ):

            if index_id < 0:
                continue

            record = dict(
                self.records[int(index_id)]
            )

            record["score"] = float(score)

            results.append(record)

        return results

    # =========================================================
    # CONTEXT BUILDING
    # =========================================================

    @staticmethod
    def _build_context(records):
        """Convert retrieved records into LLM context."""

        blocks = []

        for record in records:

            blocks.append(
                f"[{record['source']}, "
                f"p.{record['page']}]\n"
                f"{record['text']}"
            )

        return "\n\n".join(blocks)

    # =========================================================
    # OPENAI
    # =========================================================

    def _openai_answer(
        self,
        question,
        context,
        model,
    ):
        """Generate an answer using OpenAI."""

        from openai import OpenAI

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        client = OpenAI(
            api_key=api_key
        )

        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Evidence:\n\n"
                        f"{context}\n\n"
                        f"Question:\n"
                        f"{question}"
                    ),
                },
            ],
        )

        return response.choices[
            0
        ].message.content

    # =========================================================
    # GEMINI
    # =========================================================

    def _gemini_answer(
        self,
        question,
        context,
        model,
    ):
        """Generate an answer using Gemini."""

        from google import genai

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        client = genai.Client(
            api_key=api_key
        )

        prompt = (
            SYSTEM_PROMPT
            + "\n\nEvidence:\n"
            + context
            + "\n\nQuestion:\n"
            + question
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        return response.text

    # =========================================================
    # DEMO MODE
    # =========================================================

    @staticmethod
    def _demo_answer(
        question,
        records,
    ):
        """
        Deterministic fallback mode.
        No external LLM API is used.
        """

        if not records:
            return (
                "No relevant evidence was found."
            )

        lead = records[0]

        return (
            "### Demo Mode\n\n"
            "No external LLM was called.\n\n"
            "The most relevant evidence for your "
            f"question was found in "
            f"**{lead['source']} "
            f"(page {lead['page']})**.\n\n"
            "Configure an OpenAI or Gemini API key "
            "to generate a complete grounded answer."
        )

    # =========================================================
    # MAIN ANSWER FUNCTION
    # =========================================================

    def answer(
        self,
        question,
        provider,
        model,
        top_k=5,
    ):
        """Retrieve evidence and generate the final answer."""

        records = self.retrieve(
            question,
            top_k,
        )

        if not records:
            return {
                "answer": (
                    "I could not find relevant "
                    "evidence in the supplied documents."
                ),
                "sources": [],
            }

        context = self._build_context(
            records
        )

        provider = provider.strip()

        if provider == "OpenAI":

            answer = self._openai_answer(
                question=question,
                context=context,
                model=model,
            )

        elif provider == "Gemini":

            answer = self._gemini_answer(
                question=question,
                context=context,
                model=model,
            )

        else:

            answer = self._demo_answer(
                question=question,
                records=records,
            )

        sources = []

        for record in records:

            sources.append(
                {
                    "source": record["source"],
                    "page": record["page"],
                    "score": record["score"],
                    "snippet": (
                        record["text"][:500]
                        + (
                            "..."
                            if len(record["text"]) > 500
                            else ""
                        )
                    ),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }