import os
import re
from pathlib import Path

import numpy as np
import faiss
import fitz
from sentence_transformers import SentenceTransformer

SYSTEM_PROMPT = """You are a careful business and research document assistant.
Answer ONLY from the supplied evidence. If the evidence is insufficient, say so
clearly instead of inventing facts. Cite evidence using [Source, p.X].
Keep answers concise but useful. When comparing entities, use a table when helpful.
"""

class RAGEngine:
    def __init__(self):
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self.records = []
        self.index_dir = Path(os.getenv("INDEX_DIR", "data/index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _clean(self, text):
        text = re.sub(r"\s+", " ", text or "").strip()
        return text

    def _chunk(self, text, chunk_size=900, overlap=150):
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap
        return chunks

    def ingest(self, pdf_paths):
        records = []
        for pdf_path in pdf_paths:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, start=1):
                text = self._clean(page.get_text("text"))
                if not text:
                    continue
                for chunk_id, chunk in enumerate(self._chunk(text)):
                    records.append(
                        {
                            "text": chunk,
                            "source": Path(pdf_path).name,
                            "page": page_num,
                            "chunk_id": chunk_id,
                        }
                    )

        if not records:
            raise ValueError("No readable text was found in the uploaded PDFs.")

        embeddings = self.embedder.encode(
            [r["text"] for r in records],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.records = records
        faiss.write_index(index, str(self.index_dir / "vectors.faiss"))

        import json
        (self.index_dir / "records.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "documents": len(set(r["source"] for r in records)),
            "pages": len(set((r["source"], r["page"]) for r in records)),
            "chunks": len(records),
        }

    def _load(self):
        index_path = self.index_dir / "vectors.faiss"
        records_path = self.index_dir / "records.json"
        if index_path.exists() and records_path.exists():
            import json
            self.index = faiss.read_index(str(index_path))
            self.records = json.loads(records_path.read_text(encoding="utf-8"))

    def has_index(self):
        return self.index is not None and len(self.records) > 0

    def retrieve(self, question, top_k=5):
        q = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, ids = self.index.search(q, min(top_k, len(self.records)))

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            record = dict(self.records[int(idx)])
            record["score"] = float(score)
            results.append(record)
        return results

    def _build_context(self, records):
        blocks = []
        for r in records:
            blocks.append(
                f"[{r['source']}, p.{r['page']}]\n{r['text']}"
            )
        return "\n\n".join(blocks)

    def _openai_answer(self, question, context, model):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Evidence:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        return response.choices[0].message.content

    def _gemini_answer(self, question, context, model):
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = (
            SYSTEM_PROMPT
            + f"\n\nEvidence:\n{context}\n\nQuestion: {question}"
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text

    def _demo_answer(self, question, records):
        # Deterministic fallback so the application can be demonstrated
        # without an LLM API key.
        lead = records[0]
        return (
            "Demo mode is active, so no external LLM was called. "
            f"The most relevant evidence for your question was found in "
            f"**{lead['source']} (page {lead['page']})**. "
            "Add an OpenAI or Gemini API key to generate a grounded natural-language answer."
        )

    def answer(self, question, provider, model, top_k=5):
        records = self.retrieve(question, top_k)
        context = self._build_context(records)

        if provider == "OpenAI":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.")
            answer = self._openai_answer(question, context, model)
        elif provider == "Gemini":
            if not os.getenv("GEMINI_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env.")
            answer = self._gemini_answer(question, context, model)
        else:
            answer = self._demo_answer(question, records)

        sources = [
            {
                "source": r["source"],
                "page": r["page"],
                "score": r["score"],
                "snippet": r["text"][:500] + ("..." if len(r["text"]) > 500 else ""),
            }
            for r in records
        ]
        return {"answer": answer, "sources": sources}
