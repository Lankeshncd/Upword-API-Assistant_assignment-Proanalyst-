from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================
# Load PDF
# ==========================

pdf_path = "docs/upwork_api.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# ==========================
# Sanity Check
# ==========================

print("\n===== SANITY CHECK =====")
print("Character Count:", len(text))

print("\nSample Text:\n")
print(text[:500])

# ==========================
# Chunking
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("\nTotal Chunks:", len(chunks))

# ==========================
# Embedding Model
# ==========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Create FAISS Vector Store
# ==========================

db = FAISS.from_texts(
    texts=chunks,
    embedding=embedding_model
)

# ==========================
# Save Vector Store
# ==========================

db.save_local("vectorstore")

print("\n✅ Vector store created successfully!")