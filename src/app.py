"""
app.py
------
Streamlit UI for the NL2SQL pipeline.

Run with:
    streamlit run src/app.py
"""
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Allow imports from the src directory when running via `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).parent))

from nl2sql import NL2SQL  # noqa: E402  (after sys.path update)

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NL2SQL", page_icon="🗄️", layout="centered")
st.title("🗄️ NL2SQL")
st.caption("Ask a question in plain English — get back a SQL query.")

# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

index_dir = st.sidebar.text_input(
    "Index directory",
    value="artifacts/index",
    help="Directory containing schema.faiss and meta.json (built by build_index.py).",
)
top_k = st.sidebar.slider("Schema chunks to retrieve", min_value=1, max_value=10, value=5)
embedding_model = st.sidebar.selectbox(
    "Embedding model",
    options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    index=0,
)

# LLM key check
active_backend = None
for env_var, label in [
    ("GROQ_API_KEY", "Groq"),
    ("OPENAI_API_KEY", "OpenAI"),
    ("HF_API_KEY", "HuggingFace"),
]:
    if os.getenv(env_var):
        active_backend = label
        break

if active_backend:
    st.sidebar.success(f"LLM back-end: **{active_backend}**")
else:
    st.sidebar.error("No LLM API key found. Add one to your .env file.")

# ---------------------------------------------------------------------------
# Load pipeline (cached so it doesn't reload on every interaction)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading index…")
def get_pipeline(idx_dir: str, model: str, k: int) -> NL2SQL:
    return NL2SQL(index_dir=idx_dir, embedding_model=model, top_k=k)


try:
    pipeline = get_pipeline(index_dir, embedding_model, top_k)
except FileNotFoundError as e:
    st.error(
        f"**Index not found:** {e}\n\n"
        "Run `schema_chunks.py` followed by `build_index.py` to build the index first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
question = st.text_input(
    "Your question",
    placeholder="e.g. Which artists have more than 10 albums?",
)

if st.button("Generate SQL", type="primary", disabled=not question):
    if not active_backend:
        st.error("Set an LLM API key in your .env file before generating SQL.")
    else:
        with st.spinner("Retrieving schema context…"):
            chunks = pipeline.retrieve(question)

        st.subheader("Retrieved schema context")
        for chunk in chunks:
            st.markdown(
                f"**[{chunk.get('db')}.{chunk.get('table')}]** {chunk.get('text')}"
            )

        with st.spinner("Generating SQL…"):
            try:
                sql = pipeline.query(question)
            except Exception as exc:
                st.error(f"LLM call failed: {exc}")
                st.stop()

        st.subheader("Generated SQL")
        st.code(sql, language="sql")
