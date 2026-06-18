from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader(r"C:\Users\User\OneDrive\Desktop\AI_PDF_CHATBOT\uploads\python_notes.pdf.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0])