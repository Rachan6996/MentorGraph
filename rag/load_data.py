import os
import fitz  # PyMuPDF
from rag.vector_store import add_documents


def load_documents_from_folder(folder_path: str):
    """
    Reads all .txt and .pdf files and loads into FAISS+BM25
    """
    all_chunks = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
        
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(folder_path, filename))
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

    add_documents(all_chunks)
    print(f"Loaded {len(all_chunks)} chunks into FAISS+BM25")


def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text from PDF using PyMuPDF
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    
    # If no text found (scanned PDF), we could add OCR here
    # For now, return what we have
    return text


def chunk_text(text: str, chunk_size: int = 300):
    """
    Basic chunking
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks