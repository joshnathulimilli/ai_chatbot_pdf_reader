import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API Key
load_dotenv()

# Page Config
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Sidebar
with st.sidebar:

    st.title("📄 PDF Assistant")

    st.markdown("---")

    st.subheader("Technology Stack")

    st.write("🤖 Gemini")
    st.write("🔍 FAISS")
    st.write("🧠 HuggingFace")
    st.write("⚡ Streamlit")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main Title
st.title("🤖 AI PDF Assistant")
st.caption("Upload a PDF and chat with your document")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# Process PDF
if uploaded_file and not st.session_state.pdf_processed:

    with st.spinner("📚 Processing PDF..."):

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_texts(
            chunks,
            embeddings
        )

        st.session_state.db = db
        st.session_state.pdf_processed = True
        st.session_state.total_chunks = len(chunks)
        st.session_state.total_chars = len(text)

    st.success("✅ PDF Processed Successfully!")

# PDF Statistics
if st.session_state.pdf_processed:

    with st.sidebar:

        st.markdown("---")
        st.subheader("📊 PDF Statistics")

        st.write(
            f"Chunks: {st.session_state.total_chunks}"
        )

        st.write(
            f"Characters: {st.session_state.total_chars}"
        )

# Display Previous Messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input
if st.session_state.pdf_processed:

    question = st.chat_input(
        "Ask a question about the PDF..."
    )

    if question:

        # User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        # Similarity Search
        docs = st.session_state.db.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

        prompt = f"""
You are an AI PDF Assistant.

Answer ONLY from the provided context.

If the answer is not available in the PDF,
say:
"I could not find that information in the uploaded PDF."

Context:
{context}

Question:
{question}
"""

        with st.spinner("🤖 Thinking..."):

            response = llm.invoke(prompt)

            answer = response.content

        # Assistant Message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.write(answer)

else:
    st.info("📄 Upload a PDF to start chatting.")