import os
import streamlit as st
from pypdf import PdfReader
from supabase import create_client
from huggingface_hub import InferenceClient

# --- 1. Configuration & Clients ---
st.set_page_config(page_title="Personal Knowledge AI", page_icon="🧠", layout="wide")

HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
# Native feature-extraction model (384 dimensions) hosted on HF Inference API
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

@st.cache_resource
def init_clients():
    hf_client = InferenceClient(api_key=HF_TOKEN)
    sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return hf_client, sb_client

hf, supabase = init_clients()

# --- 2. Helper Functions ---
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]

def get_embedding(text: str, is_query: bool = False) -> list[float]:
    prefix = "Represent this sentence for searching relevant passages: " if is_query else ""
    vec = hf.feature_extraction(f"{prefix}{text}", model=EMBED_MODEL)
    if hasattr(vec, "flatten"):
        return vec.flatten().tolist()
    return vec[0] if isinstance(vec[0], list) else list(vec)

def ingest_document(text: str, source_name: str):
    chunks = chunk_text(text)
    rows = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk, is_query=False)
        rows.append({
            "content": chunk,
            "metadata": {"source": source_name, "chunk": i},
            "embedding": emb
        })
    if rows:
        supabase.table("documents").insert(rows).execute()
    return len(rows)

def retrieve_context(query: str, top_k: int = 4) -> list[dict]:
    query_emb = get_embedding(query, is_query=True)
    res = supabase.rpc("match_documents", {
        "query_embedding": query_emb,
        "match_threshold": 0.35,
        "match_count": top_k
    }).execute()
    return res.data or []

# --- 3. Sidebar: Knowledge Ingestion ---
with st.sidebar:
    st.header("📚 Add to Knowledge Base")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    raw_notes = st.text_area("Or paste custom notes / text:")
    source_label = st.text_input("Source label", value="personal_notes")

    if st.button("Ingest Knowledge"):
        full_text = ""
        if uploaded_pdf:
            reader = PdfReader(uploaded_pdf)
            full_text = "\n".join([p.extract_text() or "" for p in reader.pages])
            source_label = uploaded_pdf.name
        elif raw_notes.strip():
            full_text = raw_notes.strip()

        if full_text:
            with st.spinner("Embedding and storing in Supabase..."):
                try:
                    count = ingest_document(full_text, source_label)
                    st.success(f"Stored {count} chunks from '{source_label}'!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    
# --- 4. Main Chat Interface ---
st.title("🧠 Personal RAG Assistant (Qwen 2.5)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about your knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        matches = retrieve_context(prompt)
        context_block = "\n\n---\n\n".join(
            [f"[Source: {m['metadata'].get('source', 'unknown')}]\n{m['content']}" for m in matches]
        )

        system_prompt = (
            "You are a knowledgeable assistant with access to the user's personal knowledge base. "
            "Prioritize the retrieved context below when answering. If the context does not contain "
            "the answer, state that clearly before answering from your general knowledge. "
            "Respond in the same language the user writes in (e.g., Arabic or English).\n\n"
            f"RETRIEVED CONTEXT:\n{context_block if context_block else 'No matching notes found.'}"
        )

        api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-6:]

        stream = hf.chat.completions.create(
            model=LLM_MODEL,
            messages=api_messages,
            temperature=0.2,
            max_tokens=1024,
            stream=True
        )
        response_text = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)

        if matches:
            with st.expander("📄 View Retrieved Knowledge Chunks"):
                for m in matches:
                    st.caption(f"**{m['metadata'].get('source')}** (Similarity: {m['similarity']:.2f})")
                    st.write(m["content"])

    st.session_state.messages.append({"role": "assistant", "content": response_text})
