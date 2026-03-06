"""
schema_chunks.py
----------------
Reads a SQLite database, extracts table-level schema information, and writes:
  - artifacts/schemas/<db_name>.json   – full schema as JSON
  - artifacts/chunks/<db_name>.jsonl   – one line per table, used to build the FAISS index

Usage:
    python schema_chunks.py --db path/to/database.sqlite
    python schema_chunks.py --db path/to/database.sqlite --out my_artifacts --preview 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine


def open_engine(db_path: Path) -> Engine:
    """Return a SQLAlchemy engine for the given SQLite file."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    return create_engine(f"sqlite:///{db_path.as_posix()}")


def extract_schema(db_path: Path) -> Dict[str, dict]:
    """
    Inspect a SQLite database and return its schema as a dict:
        {
          "<table>": {
            "columns":      ["col_name TYPE", ...],
            "primary_keys": ["col_name", ...],
            "foreign_keys": ["col -> OtherTable(other_col)", ...]
          },
          ...
        }
    """
    eng = open_engine(db_path)
    insp = inspect(eng)

    tables = [t for t in insp.get_table_names() if not t.startswith("sqlite_")]
    schema: Dict[str, dict] = {}

    for table in tables:
        cols = insp.get_columns(table)
        pks = set(insp.get_pk_constraint(table).get("constrained_columns") or [])
        fks_raw = insp.get_foreign_keys(table)

        col_texts: List[str] = []
        for col in cols:
            entry = f"{col['name']} {col['type']}"
            if col["name"] in pks:
                entry += " PRIMARY KEY"
            col_texts.append(entry)

        fk_texts: List[str] = []
        for fk in fks_raw:
            ref_table = fk.get("referred_table")
            if not ref_table:
                continue
            left = ",".join(fk.get("constrained_columns") or [])
            right = ",".join(fk.get("referred_columns") or [])
            fk_texts.append(f"{left} -> {ref_table}({right})")

        schema[table] = {
            "columns": col_texts,
            "primary_keys": list(pks),
            "foreign_keys": fk_texts,
        }

    return schema



def make_chunks(db_name: str, schema: Dict[str, dict]) -> List[dict]:
    """
    Convert a schema dict into a list of text chunks (one per table).
    Each chunk is: {"db": <db_name>, "table": <table>, "text": <human-readable description>}
    """
    chunks: List[dict] = []
    for table, info in schema.items():
        text = f"Table: {table}. Columns: {', '.join(info['columns'])}."
        if info["foreign_keys"]:
            text += f" Foreign keys: {', '.join(info['foreign_keys'])}."
        chunks.append({"db": db_name, "table": table, "text": text})
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract schema from a SQLite database and emit table-level chunks."
    )
    ap.add_argument("--db", required=True, help="Path to .sqlite or .db file")
    ap.add_argument("--out", default="artifacts", help="Output directory (default: artifacts)")
    ap.add_argument("--preview", type=int, default=3, help="Number of chunks to print (default: 3)")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    db_name = db_path.stem

    out_dir = Path(args.out)
    schemas_dir = out_dir / "schemas"
    chunks_dir = out_dir / "chunks"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    try:
        schema = extract_schema(db_path)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not schema:
        print(f"[WARNING] No tables found in {db_path}", file=sys.stderr)
        sys.exit(0)

    chunks = make_chunks(db_name, schema)

    schema_file = schemas_dir / f"{db_name}.json"
    chunks_file = chunks_dir / f"{db_name}.jsonl"

    schema_file.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    with chunks_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"✔  {db_name}: {len(schema)} tables extracted")
    for chunk in chunks[: args.preview]:
        print("   —", chunk["text"])
    print(f"\nSaved schema → {schema_file}")
    print(f"Saved chunks → {chunks_file}")


if __name__ == "__main__":
    main()
