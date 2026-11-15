import io
import os
import json
import csv
import tempfile
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from supabase import create_client
from fastembed import TextEmbedding
from groq import Groq

# --- 1. Configuration & Clients ---
st.set_page_config(page_title="Personal Knowledge AI", page_icon="🧠", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")).strip()
RAW_SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", "")).strip()
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", "")).strip()

SUPABASE_URL = RAW_SUPABASE_URL.split("/rest/v1")[0].rstrip("/")

LLM_MODEL = "qwen/qwen3.8-27b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions

@st.cache_resource
def init_clients(groq_key: str, sb_url: str, sb_key: str):
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    groq_client = Groq(api_key=groq_key)
    sb_client = create_client(sb_url, sb_key)
    return embedder, groq_client, sb_client

embed_model, groq_client, supabase = init_clients(GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY)

# --- 2. Multi-Format File Parsing ---
def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    elif filename.endswith(".epub"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            book = epub.read_epub(tmp_path)
            chapters = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), "html.parser")
                chapters.append(soup.get_text(separator="\n", strip=True))
            return "\n\n".join(chapters)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    elif filename.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(raw_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    elif filename.endswith(".csv"):
        decoded = raw_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(decoded))
        return "\n".join([", ".join(row) for row in reader if any(row)])

    elif filename.endswith(".json"):
        decoded = raw_bytes.decode("utf-8", errors="replace")
        data = json.loads(decoded)
        return json.dumps(data, ensure_ascii=False, indent=2)

    else:
        # Default for .txt, .md, .py, .sql, etc.
        return raw_bytes.decode("utf-8", errors="replace")

# --- 3. Database & Embedding Helpers ---
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

def ingest_document(text: str, source_name: str) -> int:
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
    # Insert in batches of 100 for large books/EPUBs
    for idx in range(0, len(rows), 100):
        supabase.table("documents").insert(rows[idx:idx + 100]).execute()
    return len(rows)

def get_stored_sources() -> list[dict]:
    try:
        res = supabase.rpc("get_knowledge_sources").execute()
        return res.data or []
    except Exception:
        return []

def delete_source(source_name: str):
    supabase.table("documents").delete().eq("metadata->>source", source_name).execute()

def retrieve_context(query: str, selected_sources: list[str], top_k: int = 5) -> list[dict]:
    if not selected_sources:
        return []
    query_emb = get_embedding(query, is_query=True)
    res = supabase.rpc("match_documents", {
        "query_embedding": query_emb,
        "match_threshold": 0.30,
        "match_count": top_k,
        "filter_sources": selected_sources
    }).execute()
    return res.data or []

# --- 4. Chat History Persistence Helpers ---
def list_chat_sessions() -> list[dict]:
    try:
        res = (
            supabase.table("chat_sessions")
            .select("id, title, updated_at")
            .order("updated_at", desc=True)
            .limit(30)
            .execute()
        )
        return res.data or []
    except Exception:
        return []

def load_chat_session(session_id: str) -> list[dict]:
    res = supabase.table("chat_sessions").select("messages").eq("id", session_id).single().execute()
    return res.data.get("messages", []) if res.data else []

def save_chat_session(session_id: str | None, messages: list[dict]) -> str:
    if not messages:
        return session_id
    first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "New Chat")
    title = (first_user_msg[:40] + "...") if len(first_user_msg) > 40 else first_user_msg

    if session_id is None:
        res = supabase.table("chat_sessions").insert({
            "title": title,
            "messages": messages
        }).execute()
        return res.data[0]["id"]
    else:
        supabase.table("chat_sessions").update({
            "messages": messages,
            "updated_at": "now()"
        }).eq("id", session_id).execute()
        return session_id

def delete_chat_session(session_id: str):
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()

# --- 5. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# --- 6. Sidebar: Chat History, Source Selector & Ingestion ---
with st.sidebar:
    # A. Chat History Section
    st.header("💬 Chat History")
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()

    saved_sessions = list_chat_sessions()
    if saved_sessions:
        session_options = {s["id"]: s["title"] for s in saved_sessions}
        selected_sid = st.selectbox(
            "Previous Conversations:",
            options=[None] + list(session_options.keys()),
            format_func=lambda x: "Select a saved chat..." if x is None else session_options[x],
            index=([None] + list(session_options.keys())).index(st.session_state.current_session_id)
            if st.session_state.current_session_id in session_options else 0
        )

        if selected_sid and selected_sid != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_sid
            st.session_state.messages = load_chat_session(selected_sid)
            st.rerun()

        if st.session_state.current_session_id:
            if st.button("🗑️ Delete Current Chat", use_container_width=True):
                delete_chat_session(st.session_state.current_session_id)
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.rerun()

    st.divider()

    # B. Knowledge Base Manager & Active Filters
    st.header("🗂️ Stored Knowledge Base")
    stored_sources = get_stored_sources()
    all_source_names = [s["source"] for s in stored_sources]

    if stored_sources:
        st.caption("Available files in database:")
        for s in stored_sources:
            st.markdown(f"- **{s['source']}** (`{s['chunk_count']}` chunks)")

        selected_sources = st.multiselect(
            "Include these sources when answering:",
            options=all_source_names,
            default=all_source_names
        )

        with st.expander("Manage / Delete Files"):
            source_to_delete = st.selectbox("Select file to remove:", options=all_source_names)
            if st.button("Remove File from DB"):
                delete_source(source_to_delete)
                st.success(f"Deleted '{source_to_delete}'")
                st.rerun()
    else:
        st.info("No files stored yet. Upload files below!")
        selected_sources = []

    st.divider()

    # C. Multi-Format Ingestion Section
    st.header("📚 Add to Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Files (PDF, EPUB, DOCX, TXT, MD, CSV, JSON)",
        type=["pdf", "epub", "docx", "txt", "md", "csv", "json"],
        accept_multiple_files=True
    )
    raw_notes = st.text_area("Or paste custom notes / text:")
    custom_label = st.text_input("Label for pasted notes:", value="personal_notes")

    if st.button("Ingest Knowledge", use_container_width=True):
        if not uploaded_files and not raw_notes.strip():
            st.warning("Please upload at least one file or paste text first.")
        else:
            with st.spinner("Parsing, embedding, and storing in Supabase..."):
                try:
                    if uploaded_files:
                        for uf in uploaded_files:
                            text = extract_text_from_file(uf)
                            count = ingest_document(text, uf.name)
                            st.success(f"Stored {count} chunks from '{uf.name}'!")
                    if raw_notes.strip():
                        count = ingest_document(raw_notes.strip(), custom_label.strip() or "personal_notes")
                        st.success(f"Stored {count} chunks from '{custom_label}'!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

# --- 7. Main Chat Interface ---
st.title("🧠 Personal RAG Assistant")

if selected_sources:
    st.caption(f"🎯 **Active Knowledge Sources ({len(selected_sources)}):** {', '.join(selected_sources)}")
else:
    st.caption("⚠️ **No knowledge sources selected** — answering using general LLM knowledge only.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about your knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            matches = retrieve_context(prompt, selected_sources=selected_sources)
            context_block = "\n\n---\n\n".join(
                [f"Document: {m.get('metadata', {}).get('source', 'unknown')}\n{m.get('content', '')}" for m in matches]
            )

            system_prompt = (
                "You are a knowledgeable assistant with access to the user's personal knowledge base. "
                "Prioritize the retrieved context below when answering. Cite the document name when using "
                "information from the context. If the context does not contain the answer, state that clearly "
                "before answering from your general knowledge. Respond in the same language the user writes in "
                "(e.g., Arabic or English).\n\n"
                f"RETRIEVED CONTEXT:\n{context_block if context_block else 'No matching notes found in selected sources.'}"
            )

            api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-8:]

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
            st.session_state.current_session_id = save_chat_session(
                st.session_state.current_session_id,
                st.session_state.messages
            )
        except Exception as e:
            st.error(f"Chat / Retrieval Error: {e}")
