import hashlib
import math
import re
from collections import Counter

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI


MAX_UPLOAD_SIZE_MB = 25
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300
TOP_K_CHUNKS = 4
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


load_dotenv()

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon=":page_facing_up:",
    layout="wide",
)


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if len(word) > 2 and word not in STOP_WORDS]


def split_text(text):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def build_search_index(chunks):
    term_counts = []
    document_frequency = Counter()

    for chunk in chunks:
        counts = Counter(tokenize(chunk))
        term_counts.append(counts)
        document_frequency.update(counts.keys())

    total_chunks = len(chunks)
    idf = {
        term: math.log((1 + total_chunks) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }

    return {
        "chunks": chunks,
        "term_counts": term_counts,
        "idf": idf,
    }


def search_chunks(search_index, question, limit=TOP_K_CHUNKS):
    query_terms = tokenize(question)
    if not query_terms:
        return search_index["chunks"][:limit]

    question_lower = question.lower()
    scored_chunks = []

    for index, chunk in enumerate(search_index["chunks"]):
        counts = search_index["term_counts"][index]
        score = 0.0

        for term in query_terms:
            if term in counts:
                score += (1 + math.log(counts[term])) * search_index["idf"].get(term, 1)

        if question_lower in chunk.lower():
            score += 5

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    if not scored_chunks:
        return search_index["chunks"][:limit]

    return [chunk for _, chunk in scored_chunks[:limit]]


if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_index" not in st.session_state:
    st.session_state.search_index = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


with st.sidebar:
    st.title("AI PDF Chatbot")

    st.markdown("---")

    st.write("### Chat History")
    if st.session_state.messages:
        for index, message in enumerate(st.session_state.messages, start=1):
            role = "You" if message["role"] == "user" else "Assistant"
            preview = message["content"].strip().replace("\n", " ")
            if len(preview) > 80:
                preview = f"{preview[:77]}..."
            st.caption(f"{index}. {role}: {preview}")
    else:
        st.caption("No messages yet.")

    st.markdown("---")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.search_index = None
        st.session_state.file_hash = None
        st.rerun()


st.title("AI PDF Chatbot")
st.write("Upload a PDF and ask questions.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

if uploaded_file is not None:
    upload_size_mb = uploaded_file.size / (1024 * 1024)
    if upload_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"Please upload a PDF smaller than {MAX_UPLOAD_SIZE_MB} MB.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.file_hash != file_hash:
        st.session_state.messages = []
        st.session_state.search_index = None
        st.session_state.file_hash = file_hash

    if st.session_state.search_index is None:
        progress = st.progress(0, text="Reading PDF...")

        with st.spinner("Processing PDF..."):
            reader = PdfReader(uploaded_file)

            text_parts = []
            total_pages = len(reader.pages)
            for page_number, page in enumerate(reader.pages, start=1):
                text_parts.append(page.extract_text() or "")
                progress.progress(
                    min(page_number / max(total_pages, 1) * 0.8, 0.8),
                    text=f"Reading page {page_number} of {total_pages}...",
                )

            text = "\n".join(text_parts)

            if not text.strip():
                progress.empty()
                st.error("No readable text was found in this PDF.")
                st.stop()

            progress.progress(0.9, text="Indexing PDF...")
            chunks = split_text(text)
            st.session_state.search_index = build_search_index(chunks)

            progress.progress(1.0, text="PDF ready.")
            progress.empty()

        st.success("PDF processed successfully!")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.search_index:
    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        relevant_chunks = search_chunks(st.session_state.search_index, question)
        context = "\n\n".join(relevant_chunks)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
        )

        prompt = f"""
You are an AI PDF Assistant.

Answer ONLY using the context below.

If the answer is not available in the PDF, reply:
"I could not find that information in the uploaded PDF."

Context:
{context}

Question:
{question}
"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        answer = response.content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
else:
    st.info("Upload a PDF to start chatting.")
