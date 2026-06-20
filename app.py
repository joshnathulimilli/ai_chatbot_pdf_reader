import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


MAX_UPLOAD_SIZE_MB = 10


load_dotenv()

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon=":page_facing_up:",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


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
        st.session_state.vector_db = None
        st.rerun()


st.title("AI PDF Chatbot")
st.write("Upload a PDF and ask questions.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

if uploaded_file is not None and st.session_state.vector_db is None:
    upload_size_mb = uploaded_file.size / (1024 * 1024)
    if upload_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"Please upload a PDF smaller than {MAX_UPLOAD_SIZE_MB} MB.")
        st.stop()

    with st.spinner("Processing PDF..."):
        reader = PdfReader(uploaded_file)

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            st.error("No readable text was found in this PDF.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_text(text)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
        )

        st.session_state.vector_db = FAISS.from_texts(
            chunks,
            embeddings,
        )

    st.success("PDF processed successfully!")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.vector_db:
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

        docs = st.session_state.vector_db.similarity_search(
            question,
            k=3,
        )
        context = "\n\n".join(doc.page_content for doc in docs)

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
