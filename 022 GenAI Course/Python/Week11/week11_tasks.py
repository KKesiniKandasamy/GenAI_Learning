from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# === Step 1: Load the Document ===
file_path = "Tutorials.pdf" 

def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyMuPDFLoader(file_path)
    elif ext in [".txt", ".docx"]:
        return UnstructuredFileLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF, TXT, or DOCX.")

loader = get_loader(file_path)
documents = loader.load()

# === Step 2: Split into Chunks ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

chunks = text_splitter.split_documents(documents)

# === Step 3: Display Chunks ===
print("\n--- Displaying 3 Representative Chunks ---\n")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:\n{'-'*40}\n{chunk.page_content}\n")
