from pypdf import PdfReader

reader = PdfReader(r"C:\Users\User\OneDrive\Desktop\AI_PDF_CHATBOT\uploads\python_notes.pdf.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

print(text[:2000])