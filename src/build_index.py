"""
build_index.py
--------------
Reads a chunks.jsonl file (produced by schema_chunks.py), encodes each chunk
with a sentence-transformer model, and stores the result as a FAISS index.

Usage:
    python build_index.py --chunks_file artifacts/chunks/Chinook.jsonl --out_dir artifacts/index

Optional smoke-test:
    python build_index.py --chunks_file artifacts/chunks/Chinook.jsonl \\
                          --out_dir artifacts/index \\
                          --sample "List all employees and their orders" --k 3
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    """Normalize each row of *x* to unit length (in-place). Handles zero vectors."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    x /= norms
    return x


def load_chunks(jsonl_path: str) -> Tuple[List[Dict], List[str]]:
    """
    Read a JSONL file and return (chunks, texts).
    Empty or whitespace-only 'text' fields are skipped.
    """
    chunks: List[Dict] = []
    texts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            chunks.append(chunk)
            texts.append(text)
    return chunks, texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS cosine-similarity index from schema chunks."
    )
    parser.add_argument("--chunks_file", required=True, help="Path to <db>.jsonl produced by schema_chunks.py")
    parser.add_argument("--out_dir", required=True, help="Directory where schema.faiss and meta.json are saved")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--sample", default="List all employees and their orders", help="Smoke-test query")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbours for smoke-test (default: 3)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks, texts = load_chunks(args.chunks_file)
    if not texts:
        raise ValueError(f"No usable text found in {args.chunks_file}")
    print(f"Loaded {len(chunks)} chunks from {args.chunks_file}")

    model = SentenceTransformer(args.model)
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    embs = np.ascontiguousarray(embs, dtype=np.float32)
    print(f"Embeddings: {embs.shape}, dtype={embs.dtype}")

    embs = _normalize_l2(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    assert index.ntotal == len(chunks), "FAISS add() did not store all vectors"
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, str(out_dir / "schema.faiss"))
    (out_dir / "meta.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Saved index → {out_dir / 'schema.faiss'}")
    print(f"Saved metadata → {out_dir / 'meta.json'}")

    print(f"\n--- Smoke-test: top-{args.k} for '{args.sample}' ---")
    qv = model.encode([args.sample], convert_to_numpy=True)
    qv = np.ascontiguousarray(qv, dtype=np.float32)
    if qv.ndim == 1:
        qv = qv.reshape(1, -1)
    if qv.shape[1] != dim:
        raise ValueError(
            f"Query embedding dim ({qv.shape[1]}) does not match index dim ({dim}). "
            "Ensure the same model is used for indexing and querying."
        )
    qv = _normalize_l2(qv)
    distances, indices = index.search(qv, args.k)
    for rank, idx in enumerate(indices[0], start=1):
        ch = chunks[int(idx)]
        label = f"[{ch.get('db', '?')}.{ch.get('table', '?')}]"
        print(f"  {rank}. {label} {ch.get('text', '')}")
    print(f"  Cosine similarities: {distances[0]}")

if __name__ == "__main__":
    main()
