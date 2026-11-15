import os
import streamlit as st
from pypdf import PdfReader
from supabase import create_client
from fastembed import TextEmbedding
from groq import Groq

# --- 1. Configuration & Clients ---
st.set_page_config(page_title="Personal Knowledge AI", page_icon="🧠", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")).strip()
RAW_SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", "")).strip()
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", "")).strip()

# Guarantee clean root URL with no trailing slash or /rest/v1
SUPABASE_URL = RAW_SUPABASE_URL.split("/rest/v1")[0].rstrip("/")

# Supported Groq models: "qwen/qwen3.8-27b" | "openai/gpt-oss-120b" | "openai/gpt-oss-20b"
LLM_MODEL = "qwen/qwen3.8-27b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions

@st.cache_resource
def init_clients(groq_key: str, sb_url: str, sb_key: str):
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    groq_client = Groq(api_key=groq_key)
    sb_client = create_client(sb_url, sb_key)
    return embedder, groq_client, sb_client

embed_model, groq_client, supabase = init_clients(GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY)

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
    prefix = "query: " if is_query else "passage: "
    vec = list(embed_model.embed([f"{prefix}{text}"]))[0]
    return vec.tolist()

def ingest_document(text: str, source_name: str):
    chunks = chunk_text(text)
    if not chunks:
        return 0
    prefixed_chunks = [f"passage: {c}" for c in chunks]
    embeddings = list(embed_model.embed(prefixed_chunks))
    rows = [
        {
            "content": chunk,
            "metadata": {"source": source_name, "chunk": i},
            "embedding": emb.tolist()
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
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
        else:
            st.warning("Please upload a PDF or paste text first.")

# --- 4. Main Chat Interface ---
st.title("🧠 Personal RAG Assistant")

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
        try:
            matches = retrieve_context(prompt)
            context_block = "\n\n---\n\n".join(
                [f"Document: {m.get('metadata', {}).get('source', 'unknown')}\n{m.get('content', '')}" for m in matches]
            )

            system_prompt = (
                "You are a knowledgeable assistant with access to the user's personal knowledge base. "
                "Prioritize the retrieved context below when answering. If the context does not contain "
                "the answer, state that clearly before answering from your general knowledge. "
                "Respond in the same language the user writes in (e.g., Arabic or English).\n\n"
                f"RETRIEVED CONTEXT:\n{context_block if context_block else 'No matching notes found.'}"
            )

            api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-6:]

            stream = groq_client.chat.completions.create(
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
                        st.caption(f"**{m.get('metadata', {}).get('source', 'unknown')}** (Similarity: {m['similarity']:.2f})")
                        st.write(m["content"])

            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Chat / Retrieval Error: {e}")
