import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# Path to your folder containing PDFs
pdf_dir = "F:/interview-bot/docs"
pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# PostgreSQL pgvector connection string
CONNECTION_STRING = "postgresql+psycopg2://postgres:baargavi123@localhost:5432/vector_db"

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Container for all document chunks
all_chunks = []

# Process PDFs
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    # Add filename as metadata for each chunk
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(pdf_path)

    # Print sample output
    print(f"\n Processed: {os.path.basename(pdf_path)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}:\n{chunk.page_content[:300]}\n{'-'*50}")

    all_chunks.extend(chunks)

# Store chunks into PostgreSQL using pgvector
try:
    PGVector.from_documents(
        documents=all_chunks,
        embedding=embedding,
        collection_name="pdf_vectors",
        connection_string=CONNECTION_STRING,
    )
    print(" PDFs embedded and saved to pgvector (PostgreSQL)")
except Exception as e:
    print(" Error saving to pgvector:", e)
