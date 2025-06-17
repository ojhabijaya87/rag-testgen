from dotenv import load_dotenv
import os

load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

def chunk_text(text):
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i+CHUNK_SIZE]
        chunks.append(chunk)
    return chunks
