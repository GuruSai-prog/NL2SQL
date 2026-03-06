# NL2SQL

Convert natural-language questions into SQL queries using a retrieval-augmented generation (RAG) pipeline.

Given a question like *"Which artists have more than 10 albums?"*, the system retrieves the relevant database tables from a FAISS vector index, then sends the schema context together with the question to an LLM that writes the SQL.

---

## How it works

```
SQLite DB
    │
    ▼
schema_chunks.py   ← extracts table/column info, writes chunks.jsonl
    │
    ▼
build_index.py     ← embeds chunks with sentence-transformers, stores in FAISS
    │
    ▼
FAISS index  ◄──── nl2sql.py / app.py
                      │  1. embed the user question
                      │  2. retrieve top-k schema chunks
                      │  3. build prompt = schema + question
                      │  4. call LLM → return SQL
```

Supported databases (any SQLite file works):
- [Chinook](https://github.com/lerocha/chinook-database) – music store
- [Northwind](https://github.com/jpwhite3/northwind-SQLite3) – fictional retailer
- [Spider](https://yale-lily.github.io/spider) – academic benchmark

Supported LLM back-ends (auto-selected based on available API keys):
- **Groq** – fast, free tier available (`GROQ_API_KEY`)
- **OpenAI** – GPT-3.5 / GPT-4 (`OPENAI_API_KEY`)
- **HuggingFace Inference API** – any text-generation model (`HF_API_KEY`)

---

## Project structure

```
NL2SQL/
├── src/
│   ├── schema_chunks.py   # Step 1 – extract schema and write chunks
│   ├── build_index.py     # Step 2 – build the FAISS vector index
│   ├── nl2sql.py          # Step 3 – retrieval + LLM inference pipeline
│   └── app.py             # Optional Streamlit web UI
├── requirements.txt
└── NL2SQL.html            # Architecture diagram (open in a browser)
```

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/GuruSai-prog/NL2SQL.git
cd NL2SQL
pip install -r requirements.txt
```

**2. Create a `.env` file** in the project root with at least one LLM API key:

```
# .env
GROQ_API_KEY=your_groq_key_here
# OPENAI_API_KEY=your_openai_key_here
# HF_API_KEY=your_huggingface_key_here
```

---

## Usage

### Step 1 – Extract schema from your database

```bash
python src/schema_chunks.py --db path/to/Chinook.sqlite
# Outputs:
#   artifacts/schemas/Chinook.json
#   artifacts/chunks/Chinook.jsonl
```

Run once per database. Add `--out my_dir` to change the output folder.

### Step 2 – Build the FAISS index

```bash
python src/build_index.py \
    --chunks_file artifacts/chunks/Chinook.jsonl \
    --out_dir artifacts/index
# Outputs:
#   artifacts/index/schema.faiss
#   artifacts/index/meta.json
```

### Step 3 – Ask a question (CLI)

```bash
python src/nl2sql.py \
    --index_dir artifacts/index \
    --question "Which artists have more than 10 albums?"
```

Example output:

```
Question: Which artists have more than 10 albums?

Retrieved schema context:
  [Chinook.Artist] Table: Artist. Columns: ArtistId INTEGER PRIMARY KEY, Name NVARCHAR(120).
  [Chinook.Album]  Table: Album.  Columns: AlbumId INTEGER PRIMARY KEY, Title NVARCHAR(160), ArtistId INTEGER. Foreign keys: ArtistId -> Artist(ArtistId).

Generating SQL...

Generated SQL:
SELECT ar.Name, COUNT(al.AlbumId) AS album_count
FROM Artist ar
JOIN Album al ON ar.ArtistId = al.ArtistId
GROUP BY ar.ArtistId
HAVING COUNT(al.AlbumId) > 10;
```

### Step 3 (alternative) – Web UI

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Using as a library

```python
from src.nl2sql import NL2SQL

pipeline = NL2SQL(index_dir="artifacts/index", top_k=5)
sql = pipeline.query("List all customers from Germany")
print(sql)
```

---

## Using multiple databases

Run `schema_chunks.py` and then `build_index.py` for each database separately.  
If you want to search across all of them at once, concatenate the `.jsonl` files before indexing:

```bash
cat artifacts/chunks/*.jsonl > artifacts/chunks/all.jsonl
python src/build_index.py \
    --chunks_file artifacts/chunks/all.jsonl \
    --out_dir artifacts/index
```

---

## Configuration reference

| Environment variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | one of these | Groq API key |
| `OPENAI_API_KEY` | one of these | OpenAI API key |
| `HF_API_KEY` | one of these | HuggingFace API key |
| `HF_MODEL_ID` | no | HuggingFace model to use (default: `google/flan-t5-large`) |

| CLI flag | Script | Default | Description |
|---|---|---|---|
| `--db` | `schema_chunks.py` | — | Path to SQLite file |
| `--out` | `schema_chunks.py` | `artifacts` | Output directory |
| `--chunks_file` | `build_index.py` | — | Path to `.jsonl` produced by schema_chunks |
| `--out_dir` | `build_index.py` | — | Where to save index files |
| `--model` | `build_index.py` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `--index_dir` | `nl2sql.py` | `artifacts/index` | Directory with FAISS index |
| `--question` | `nl2sql.py` | — | Natural-language question |
| `--top_k` | `nl2sql.py` | `5` | Schema chunks to retrieve |

---
