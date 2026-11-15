import io
import os
import re
import gc
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

# --- 1. Page Config & Secrets ---
st.set_page_config(page_title="Personal Knowledge AI", page_icon="🧠", layout="wide")

APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", "")).strip()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")).strip()
RAW_SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", "")).strip()
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", "")).strip()
SUPABASE_URL = RAW_SUPABASE_URL.split("/rest/v1")[0].rstrip("/")

# --- Priority 1: Password Authentication Gate ---
if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🔒 Personal Knowledge AI — Login")
        pwd = st.text_input("Enter your APP_PASSWORD to unlock:", type="password")
        if st.button("Unlock App"):
            if pwd == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# --- 2. Cached Clients & Model Loader ---
@st.cache_resource
def init_core_clients(groq_key: str, sb_url: str, sb_key: str):
    return Groq(api_key=groq_key), create_client(sb_url, sb_key)

@st.cache_resource
def load_embedder(model_name: str):
    return TextEmbedding(model_name=model_name)

groq_client, supabase = init_core_clients(GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY)

# --- 3. Multi-Format & Smart WhatsApp Parsing (Priority 5) ---
WHATSAPP_LINE_RE = re.compile(
    r"^(\[?\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm]|[\s\u202f]*[صم])?\]?\s*(?:-|\~)?\s*)"
)

def clean_whatsapp_and_noise(text: str) -> str:
    text = re.sub(r"<Media omitted>|<تم استبعاد الوسائط>|‎", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_whatsapp_by_turns(lines: list[str], max_chars: int = 2200, overlap_msgs: int = 3) -> list[str]:
    messages = []
    current_msg = []
    for line in lines:
        if WHATSAPP_LINE_RE.match(line):
            if current_msg:
                messages.append("\n".join(current_msg).strip())
            current_msg = [line]
        else:
            current_msg.append(line)
    if current_msg:
        messages.append("\n".join(current_msg).strip())

    chunks = []
    window = []
    window_len = 0
    for msg in messages:
        if not msg:
            continue
        if window_len + len(msg) > max_chars and window:
            chunks.append("\n".join(window))
            window = window[-overlap_msgs:]
            window_len = sum(len(m) for m in window)
        window.append(msg)
        window_len += len(msg) + 1
    if window:
        chunks.append("\n".join(window))
    return [c for c in chunks if len(c) > 30]

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = clean_whatsapp_and_noise(text)
    lines = text.splitlines()
    # Auto-detect WhatsApp logs if >= 5 of the first 40 lines match WhatsApp timestamps
    sample_matches = sum(1 for l in lines[:40] if WHATSAPP_LINE_RE.match(l))
    if sample_matches >= 5:
        return chunk_whatsapp_by_turns(lines)

    if len(text) > 1_000_000:
        chunk_size = 2400
        overlap = 200

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

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
        return json.dumps(json.loads(decoded), ensure_ascii=False, indent=2)
    else:
        return raw_bytes.decode("utf-8", errors="replace")

# --- 4. Database, Hybrid Search & Query Rewriting (Priorities 2, 3, 4) ---
def get_embedding(text: str, embed_model_name: str, is_query: bool = False) -> list[float]:
    embedder = load_embedder(embed_model_name)
    prefix = ""
    if "bge" in embed_model_name.lower():
        prefix = "query: " if is_query else "passage: "
    vec = list(embedder.embed([f"{prefix}{text}"]))[0]
    return vec.tolist()

def delete_source(source_name: str):
    supabase.table("documents").delete().eq("metadata->>source", source_name).execute()

def ingest_document(text: str, source_name: str, embed_model_name: str) -> int:
    embedder = load_embedder(embed_model_name)
    use_prefix = "bge" in embed_model_name.lower()
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    if total_chunks == 0:
        return 0

    delete_source(source_name)
    batch_size = 64
    progress_bar = st.progress(0, text=f"Starting ingestion of {total_chunks:,} chunks...")

    for start_idx in range(0, total_chunks, batch_size):
        batch_chunks = chunks[start_idx : start_idx + batch_size]
        prefixed_batch = [f"passage: {c}" if use_prefix else c for c in batch_chunks]
        batch_embeddings = list(embedder.embed(prefixed_batch))

        rows = [
            {
                "content": chunk,
                "metadata": {"source": source_name, "chunk": start_idx + i, "model": embed_model_name},
                "embedding": emb.tolist()
            }
            for i, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings))
        ]
        supabase.table("documents").insert(rows).execute()

        del batch_embeddings, rows, prefixed_batch
        gc.collect()

        done = min(start_idx + batch_size, total_chunks)
        progress_bar.progress(done / total_chunks, text=f"Embedded & stored {done:,} / {total_chunks:,} chunks...")

    progress_bar.empty()
    del chunks
    gc.collect()
    return total_chunks

def rewrite_followup_query(user_prompt: str, chat_history: list[dict], llm_model: str) -> str:
    if len(chat_history) < 2:
        return user_prompt
    recent_turns = "\n".join([f"{m['role']}: {m['content'][:300]}" for m in chat_history[-4:]])
    try:
        res = groq_client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given the recent conversation, rewrite the user's latest input into a concise, "
                        "standalone search query (in the same language) suitable for vector and keyword retrieval. "
                        "Output ONLY the rewritten query text and nothing else."
                    )
                },
                {"role": "user", "content": f"Recent Chat:\n{recent_turns}\n\nLatest Input: {user_prompt}"}
            ],
            temperature=0.0,
            max_tokens=80
        )
        rewritten = res.choices[0].message.content.strip()
        return rewritten if rewritten else user_prompt
    except Exception:
        return user_prompt

def retrieve_context(
    query: str,
    selected_sources: list[str],
    embed_model_name: str,
    top_k: int = 5,
    match_threshold: float = 0.28
) -> list[dict]:
    if not selected_sources:
        return []
    query_emb = get_embedding(query, embed_model_name, is_query=True)
    res = supabase.rpc("match_documents", {
        "query_embedding": query_emb,
        "match_threshold": match_threshold,
        "match_count": top_k,
        "filter_sources": selected_sources,
        "query_text": query
    }).execute()
    return res.data or []

def get_stored_sources() -> list[dict]:
    try:
        res = supabase.rpc("get_knowledge_sources").execute()
        return res.data or []
    except Exception:
        return []

# --- 5. Chat History Helpers ---
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
        res = supabase.table("chat_sessions").insert({"title": title, "messages": messages}).execute()
        return res.data[0]["id"]
    else:
        supabase.table("chat_sessions").update({"messages": messages, "updated_at": "now()"}).eq("id", session_id).execute()
        return session_id

def delete_chat_session(session_id: str):
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()

# --- 6. Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# --- 7. Sidebar Controls (Priorities 3, 6, 7) ---
with st.sidebar:
    # A. Chat History
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

    # B. Knowledge Base Manager & Health Status (Priority 7)
    st.header("🗂️ Stored Knowledge Base")
    stored_sources = get_stored_sources()
    all_source_names = [s["source"] for s in stored_sources]
    total_db_chunks = sum(int(s.get("chunk_count", 0)) for s in stored_sources)

    st.caption(f"🟢 **Supabase Active** — `{total_db_chunks:,}` total chunks (~{total_db_chunks * 0.007:.1f} MB / 500 MB)")

    if stored_sources:
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

    # C. Model & Retrieval Tuning Controls (Priorities 3 & 6)
    with st.expander("⚙️ AI & Retrieval Settings", expanded=False):
        llm_model = st.selectbox(
            "Groq LLM Model (switch if rate-limited):",
            options=[
                "qwen/qwen3.8-27b",
                "qwen/qwen3.6-27b",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b"
            ],
            index=0
        )
        embed_model_choice = st.selectbox(
            "Embedding Model (384-dim):",
            options=[
                "BAAI/bge-small-en-v1.5",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=0,
            help="Keep BAAI/bge-small-en-v1.5 for already-uploaded chunks, or switch to Multilingual MiniLM before re-uploading Arabic/multilingual files."
        )
        top_k_slider = st.slider("Chunks to Retrieve (Top-K):", min_value=2, max_value=10, value=5)
        threshold_slider = st.slider("Similarity Threshold:", min_value=0.10, max_value=0.70, value=0.28, step=0.02)
        enable_query_rewrite = st.checkbox("Contextual Follow-Up Query Rewriting", value=True)

    st.divider()

    # D. Multi-Format Ingestion
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
                            count = ingest_document(text, uf.name, embed_model_choice)
                            st.success(f"Stored {count:,} chunks from '{uf.name}'!")
                    if raw_notes.strip():
                        count = ingest_document(raw_notes.strip(), custom_label.strip() or "personal_notes", embed_model_choice)
                        st.success(f"Stored {count:,} chunks from '{custom_label}'!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

# --- 8. Main Chat Interface ---
st.title("🧠 Personal RAG Assistant")

if selected_sources:
    st.caption(f"🎯 **Active Sources ({len(selected_sources)}):** {', '.join(selected_sources)} | 🤖 **Model:** `{llm_model}`")
else:
    st.caption("⚠️ **No knowledge sources selected** — answering using general LLM knowledge only.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about your knowledge base..."):
    history_before_prompt = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            search_query = (
                rewrite_followup_query(prompt, history_before_prompt, llm_model)
                if enable_query_rewrite and history_before_prompt
                else prompt
            )

            matches = retrieve_context(
                search_query,
                selected_sources=selected_sources,
                embed_model_name=embed_model_choice,
                top_k=top_k_slider,
                match_threshold=threshold_slider
            )
            context_block = "\n\n---\n\n".join(
                [f"Document: {m.get('metadata', {}).get('source', 'unknown')}\n{m.get('content', '')}" for m in matches]
            )

            system_prompt = (
                "You are a knowledgeable assistant with access to the user's personal knowledge base. "
                "Prioritize the retrieved context below when answering. Cite the document name when using "
                "information from the context. If the context does not contain the answer, state that clearly "
                "before answering from your general knowledge. Respond in the same language the user writes in "
                "(e.g., Egyptian Arabic or English).\n\n"
                f"RETRIEVED CONTEXT:\n{context_block if context_block else 'No matching notes found in selected sources.'}"
            )

            api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-8:]

            stream = groq_client.chat.completions.create(
                model=llm_model,
                messages=api_messages,
                temperature=0.2,
                max_tokens=1024,
                stream=True
            )
            response_text = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)

            if matches:
                with st.expander(f"📄 View Retrieved Chunks (Search query: '{search_query}')"):
                    for m in matches:
                        st.caption(f"**{m.get('metadata', {}).get('source', 'unknown')}** (Hybrid Score: {m['similarity']:.2f})")
                        st.write(m["content"])

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.current_session_id = save_chat_session(
                st.session_state.current_session_id,
                st.session_state.messages
            )
        except Exception as e:
            st.error(f"Chat / Retrieval Error: {e}")
