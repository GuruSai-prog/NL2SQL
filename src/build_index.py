import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse
from typing import List, Dict

def safe_normalize_L2(x: np.ndarray) -> np.ndarray:
    """Normalize rows of x to unit length without creating NaNs."""
    # x: (n, d) float32 contiguous
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # avoid divide-by-zero (blank chunks)
    norms = np.where(norms == 0.0, 1.0, norms)
    x /= norms
    return x

def load_chunks(jsonl_path: str):
    chunks: List[Dict] = []
    texts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            ch = json.loads(line)
            t = (ch.get("text") or "").strip()
            if not t:        # skip empty/whitespace rows
                continue
            chunks.append(ch)
            texts.append(t)
    return chunks, texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_file", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--out_dir", required=True, help="Where to save FAISS index & metadata")
    parser.add_argument("--sample", default="List all employees and their orders")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load chunks/texts
    chunks, texts = load_chunks(args.chunks_file)
    if len(texts) == 0:
        raise ValueError("No non-empty texts found in chunks file.")
    print(f"Loaded {len(chunks)} chunks (non-empty) from {args.chunks_file}")

    # 2) Encode
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    embs = np.asarray(embs, dtype=np.float32, order="C")  # contiguous float32
    print(f"Embeddings shape: {embs.shape}, dtype: {embs.dtype}")

    # 3) Cosine via IP on unit vectors
    embs = safe_normalize_L2(embs)

    # 4) Build index
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine when vectors are normalized
    index.add(embs) # type: ignore
    assert index.ntotal == embs.shape[0], "FAISS add() did not add all vectors"
    print("FAISS index size:", index.ntotal)

    # 5) Persist
    faiss.write_index(index, str(out_dir / "schema.faiss"))
    (out_dir / "meta.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print("Saved FAISS index and metadata to", out_dir)

    # 6) Sanity search
    qv = model.encode([args.sample], convert_to_numpy=True)
    qv = np.asarray(qv, dtype=np.float32, order="C")
    if qv.ndim == 1:
        qv = qv.reshape(1, -1)
    if qv.shape[1] != d:
        raise ValueError(f"Query dim {qv.shape[1]} != index dim {d}. "
                         "Check you used the same embedding model.")
    qv = safe_normalize_L2(qv)

    D, I = index.search(qv, args.k)  # pyright: ignore[reportCallIssue] # qv: (1, d)
    print("\nSample query:", args.sample)
    for rank, idx in enumerate(I[0], start=1):
        ch = chunks[int(idx)]
        prefix = f"[{ch.get('db','?')}.{ch.get('table','?')}]"
        print(f"{rank}. {prefix} {ch.get('text','')}")
    print("\nDistances (cosine similarity):", D[0])

if __name__ == "__main__":
    main()
