from pypdf import PdfReader
from docx import Document
import os

def load_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])