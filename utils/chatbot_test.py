from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

load_dotenv()

# Read PDF
reader = PdfReader(r"C:\Users\User\OneDrive\Desktop\AI_PDF_CHATBOT\uploads\python_notes.pdf.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text() or ""

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS
db = FAISS.from_texts(chunks, embeddings)

# User Question
question = "What is Python?"

docs = db.similarity_search(question, k=3)

context = "\n".join([doc.page_content for doc in docs])

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

prompt = f"""
Answer using the context below.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nANSWER:\n")
print(response.content)