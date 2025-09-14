from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

def open_engine(db_path: Path) -> Engine:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    return create_engine(f"sqlite:///{db_path.as_posix()}")

def extract_schema(db_path: Path) -> Dict[str, dict]:
    """
    Returns {table: {columns:[..], primary_keys:[..], foreign_keys:[..]}}
    - columns: 'Name TYPE' (+ ' PRIMARY KEY' on PK columns)
    - foreign_keys: 'colA,colB -> OtherTable(OtherColA,OtherColB)'
    """
    eng = open_engine(db_path)
    insp = inspect(eng)

    tables = [t for t in insp.get_table_names() if not t.startswith("sqlite_")]
    schema: Dict[str, dict] = {}

    for t in tables:
        cols = insp.get_columns(t)
        pks = set(insp.get_pk_constraint(t).get("constrained_columns") or [])
        fks_raw = insp.get_foreign_keys(t)

        col_texts: List[str] = []
        for c in cols:
            txt = f"{c['name']} {c['type']}"
            if c["name"] in pks:
                txt += " PRIMARY KEY"
            col_texts.append(txt)

        fk_texts: List[str] = []
        for fk in fks_raw:
            ref_table = fk.get("referred_table")
            if not ref_table:
                continue
            left = ",".join(fk.get("constrained_columns") or [])
            right = ",".join(fk.get("referred_columns") or [])
            fk_texts.append(f"{left} -> {ref_table}({right})")

        schema[t] = {
            "columns": col_texts,
            "primary_keys": list(pks),
            "foreign_keys": fk_texts,
        }
    return schema

def make_chunks(db_name: str, schema: Dict[str, dict]) -> List[dict]:
    """One chunk per table → {'db','table','text'}"""
    chunks: List[dict] = []
    for t, info in schema.items():
        txt = f"Table: {t}. Columns: {', '.join(info['columns'])}."
        if info["foreign_keys"]:
            txt += f" Foreign keys: {', '.join(info['foreign_keys'])}."
        chunks.append({"db": db_name, "table": t, "text": txt})
    return chunks

def main():
    ap = argparse.ArgumentParser(description="Extract schema & emit table-level chunks")
    ap.add_argument("--db", required=True, help="Path to .sqlite/.db file")
    ap.add_argument("--out", default="artifacts", help="Output dir (default: artifacts)")
    ap.add_argument("--preview", type=int, default=3, help="How many chunks to print")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    db_name = db_path.stem  # e.g., Chinook / northwind / academic

    out_dir = Path(args.out)
    (out_dir / "schemas").mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)

    try:
        schema = extract_schema(db_path)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    chunks = make_chunks(db_name, schema)

    # Save files
    (out_dir / "schemas" / f"{db_name}.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    with (out_dir / "chunks" / f"{db_name}.jsonl").open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"✔ {db_name}: {len(schema)} tables")
    for ch in chunks[: args.preview]:
        print("—", ch["text"])
    print("Saved:",
          out_dir / "schemas" / f"{db_name}.json",
          "and",
          out_dir / "chunks" / f"{db_name}.jsonl")

if __name__ == "__main__":
    main()
