from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

def load_retriever():
    persist_dir = os.getenv("CHROMA_DIR", "vectorstore/chroma")
    embeddings = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db.as_retriever()
