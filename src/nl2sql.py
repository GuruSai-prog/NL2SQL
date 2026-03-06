"""
nl2sql.py
---------
Core NL-to-SQL pipeline.

Steps
-----
1. Load the FAISS index + chunk metadata built by build_index.py.
2. Encode the user's natural-language question.
3. Retrieve the top-k most relevant schema chunks (tables).
4. Build a prompt that includes the retrieved schema context.
5. Call an LLM to generate the SQL query.

Supported LLM back-ends (set via environment variables):
  - Groq  (GROQ_API_KEY)        – fast, free tier available
  - OpenAI (OPENAI_API_KEY)     – GPT-3.5/GPT-4
  - HuggingFace (HF_API_KEY)    – inference API

The back-end is auto-selected in the order above; the first one whose key
is present in the environment is used.

Usage (CLI):
    python nl2sql.py --index_dir artifacts/index \
                     --question "Which artists have more than 10 albums?"

Usage (as a library):
    from nl2sql import NL2SQL
    pipeline = NL2SQL(index_dir="artifacts/index")
    sql = pipeline.query("Which artists have more than 10 albums?")
    print(sql)
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    x /= norms
    return x


def load_index(index_dir: str | Path) -> Tuple[faiss.Index, List[Dict]]:
    """Load schema.faiss and meta.json from *index_dir*."""
    d = Path(index_dir)
    faiss_path = d / "schema.faiss"
    meta_path = d / "meta.json"
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    index = faiss.read_index(str(faiss_path))
    chunks: List[Dict] = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, chunks


_SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database schema and a natural-language question, "
    "write a single, correct SQL SELECT query that answers the question. "
    "Return only the SQL — no explanation, no markdown fences."
)


def build_prompt(question: str, schema_chunks: List[Dict]) -> str:
    """Combine retrieved schema chunks and the user question into a prompt."""
    schema_lines = []
    for chunk in schema_chunks:
        schema_lines.append(chunk.get("text", ""))
    schema_block = "\n".join(schema_lines)

    return textwrap.dedent(f"""\
        ### Database schema (relevant tables)
        {schema_block}

        ### Question
        {question}

        ### SQL
    """)


def _call_groq(prompt: str, system: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _call_openai(prompt: str, system: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _call_hf(prompt: str, system: str) -> str:
    """Calls the HuggingFace Inference API (text-generation endpoint)."""
    import requests
    full_prompt = f"{system}\n\n{prompt}"
    headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}
    payload = {
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.01},
    }
    model_id = os.getenv("HF_MODEL_ID", "google/flan-t5-large")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return (data[0].get("generated_text") or "").strip()
    raise RuntimeError(f"Unexpected HF response: {data}")


def call_llm(prompt: str, system: str = _SYSTEM_PROMPT) -> str:
    """Auto-select an LLM back-end and return its text response."""
    if os.getenv("GROQ_API_KEY"):
        return _call_groq(prompt, system)
    if os.getenv("OPENAI_API_KEY"):
        return _call_openai(prompt, system)
    if os.getenv("HF_API_KEY"):
        return _call_hf(prompt, system)
    raise EnvironmentError(
        "No LLM API key found. Set one of: GROQ_API_KEY, OPENAI_API_KEY, HF_API_KEY."
    )


class NL2SQL:
    """
    End-to-end NL-to-SQL pipeline.

    Parameters
    ----------
    index_dir : str | Path
        Directory containing schema.faiss and meta.json.
    embedding_model : str
        Sentence-Transformers model used when building the index.
        Must match the model passed to build_index.py (default: all-MiniLM-L6-v2).
    top_k : int
        Number of schema chunks to retrieve for each query.
    """

    def __init__(
        self,
        index_dir: str | Path = "artifacts/index",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self.index, self.chunks = load_index(index_dir)
        self.embed_model = SentenceTransformer(embedding_model)
        self._dim = self.index.d

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Dict]:
        """Return the top-k schema chunks most relevant to *question*."""
        k = k or self.top_k
        qv = self.embed_model.encode([question], convert_to_numpy=True)
        qv = np.ascontiguousarray(qv, dtype=np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        if qv.shape[1] != self._dim:
            raise ValueError(
                f"Query embedding dim {qv.shape[1]} != index dim {self._dim}. "
                "Use the same embedding model as during indexing."
            )
        qv = _normalize_l2(qv)
        _, indices = self.index.search(qv, k)
        return [self.chunks[int(i)] for i in indices[0] if 0 <= int(i) < len(self.chunks)]

    def query(self, question: str, k: Optional[int] = None) -> str:
        """
        Full pipeline: retrieve relevant schema → build prompt → call LLM.

        Returns the generated SQL string.
        """
        context_chunks = self.retrieve(question, k)
        prompt = build_prompt(question, context_chunks)
        return call_llm(prompt)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Translate a natural-language question to SQL.")
    parser.add_argument("--index_dir", default="artifacts/index", help="Directory with schema.faiss and meta.json")
    parser.add_argument("--question", required=True, help="Natural-language question to translate")
    parser.add_argument("--top_k", type=int, default=5, help="Number of schema chunks to retrieve (default: 5)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model (must match build_index.py)")
    args = parser.parse_args()

    pipeline = NL2SQL(index_dir=args.index_dir, embedding_model=args.model, top_k=args.top_k)

    print(f"\nQuestion: {args.question}\n")
    print("Retrieved schema context:")
    for chunk in pipeline.retrieve(args.question):
        print(f"  [{chunk.get('db')}.{chunk.get('table')}] {chunk.get('text')}")

    print("\nGenerating SQL...")
    sql = pipeline.query(args.question)
    print("\nGenerated SQL:")
    print(sql)

if __name__ == "__main__":
    main()
