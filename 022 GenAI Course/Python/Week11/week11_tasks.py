from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# === Step 3: Generate Embeddings ===
embedding_model = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl"
)

# === Step 4: Store in FAISS Vector DB ===
vectorstore = FAISS.from_documents(chunks, embedding_model)

# === Step 5: Semantic Search ===
query = "What is the company's leave policy?"
results = vectorstore.similarity_search(query, k=3)

# === Step 6: Display Results ===
print(f"\n🔍 Semantic Search Query:\n{query}\n")
print("--- Top 3 Relevant Chunks ---\n")
for i, res in enumerate(results):
    print(f"Result {i+1}:\n{'-'*40}\n{res.page_content}\n")

# === Step 7: Explanation ===
print("\n🧠 Embedding & Vector Search Setup:")
print("""
• Used HuggingFaceInstructorEmbeddings ('hkunlp/instructor-xl') for high-quality semantic vector representations.
• Stored vectors in FAISS, a fast and scalable similarity search library.
• Executed semantic search using cosine similarity to retrieve the top 3 most relevant chunks.
""")