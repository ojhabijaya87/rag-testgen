from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

def embed_and_store(chunks):
    persist_dir = os.getenv("CHROMA_DIR", "vectorstore/chroma")
    embeddings = HuggingFaceEmbeddings()
    Chroma.from_texts(chunks, embeddings, persist_directory=persist_dir)
