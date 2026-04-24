import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import streamlit as st
import torch
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_TITLE = "Local Document RAG Chat"
DEFAULT_LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

@dataclass
class Chunk:
    text: str
    source_name: str
    chunk_id: int

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"\n[Page {page_num}]\n{text}")
    return "\n".join(parts).strip()

def read_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = Document(tmp_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

def read_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix == ".pdf":
        return read_pdf(data)
    if suffix == ".docx":
        return read_docx(data)
    if suffix in {".txt", ".md", ".py", ".csv", ".json"}:
        return read_text(data)
    raise ValueError(f"Unsupported file type: {suffix}")

def chunk_text(text: str, source_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, source_name=source_name, chunk_id=idx))
            idx += 1
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

def build_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index

def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    embs = embedder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embs.astype("float32")

def retrieve(query: str, embedder, index, chunks: List[Chunk], top_k: int = 4):
    q_emb = embed_texts(embedder, [query])
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"score": float(score), "chunk": chunks[int(idx)]})
    return results

def build_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        chunk = item["chunk"]
        context_blocks.append(f"[Source {i}: {chunk.source_name} | chunk {chunk.chunk_id}]\n{chunk.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."
    system = (
        "You are a careful assistant answering questions only from the provided document context. "
        "If the answer is not in the context, say you don't know based on the uploaded documents. "
        "Cite sources inline like [Source 1]."
    )
    return f"""<|system|>
{system}
<|user|>
Context:
{context}

Question:
{question}

Answer using only the context above.
<|assistant|>
"""

def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 400, temperature: float = 0.2) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def reset_index():
    for key in ("chunks", "index", "doc_stats"):
        if key in st.session_state:
            del st.session_state[key]

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Runs locally with public Hugging Face models. No API key required.")

with st.sidebar:
    st.header("Settings")
    llm_name = st.text_input("LLM model", value=DEFAULT_LLM)
    embed_model_name = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    top_k = st.slider("Retrieved chunks", 2, 8, 4)
    max_new_tokens = st.slider("Max new tokens", 128, 1024, 400, 32)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    if st.button("Clear index"):
        reset_index()
        st.success("Cleared indexed documents.")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "md", "py", "csv", "json"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Build / Rebuild index", type="primary"):
    all_chunks = []
    doc_stats = []
    with st.spinner("Reading files and building embeddings..."):
        embedder = load_embedder(embed_model_name)
        for file in uploaded_files:
            try:
                text = extract_text(file)
                chunks = chunk_text(text, file.name)
                all_chunks.extend(chunks)
                doc_stats.append({"file": file.name, "chars": len(text), "chunks": len(chunks)})
            except Exception as exc:
                st.error(f"Failed to process {file.name}: {exc}")
        if all_chunks:
            embeddings = embed_texts(embedder, [c.text for c in all_chunks])
            st.session_state["index"] = build_faiss_index(embeddings)
            st.session_state["chunks"] = all_chunks
            st.session_state["doc_stats"] = doc_stats
            st.success(f"Indexed {len(all_chunks)} chunks from {len(doc_stats)} file(s).")
        else:
            st.warning("No text could be indexed.")

if "doc_stats" in st.session_state:
    with st.expander("Indexed files", expanded=False):
        st.table(st.session_state["doc_stats"])

question = st.chat_input("Ask a question about your uploaded documents")

if question:
    if "index" not in st.session_state or "chunks" not in st.session_state:
        st.error("Upload files and click 'Build / Rebuild index' first.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                embedder = load_embedder(embed_model_name)
                tokenizer, model = load_llm(llm_name)
                retrieved = retrieve(question, embedder, st.session_state["index"], st.session_state["chunks"], top_k)
                prompt = build_prompt(question, retrieved)
                answer = generate_answer(tokenizer, model, prompt, max_new_tokens, temperature)
                st.write(answer if answer else "I couldn't generate an answer.")
            with st.expander("Retrieved context", expanded=False):
                for i, item in enumerate(retrieved, start=1):
                    chunk = item["chunk"]
                    st.markdown(f"**Source {i}** — `{chunk.source_name}` chunk `{chunk.chunk_id}` | score `{item['score']:.3f}`")
                    st.write(chunk.text)
                    st.markdown("---")

st.markdown("---")
st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")