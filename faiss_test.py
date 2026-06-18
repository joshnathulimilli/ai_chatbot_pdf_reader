from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
reader = PdfReader(r"C:\Users\User\OneDrive\Desktop\AI_PDF_CHATBOT\uploads\python_notes.pdf.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

print("PDF Loaded Successfully!")
print("Total Characters:", len(text))

# Split Text into Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

# Create Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating FAISS Index...")

# Create FAISS Vector Store
db = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)

# Save Index
db.save_local("data/faiss_index")

print("FAISS Index Created Successfully!")
print("Saved in: data/faiss_index")
