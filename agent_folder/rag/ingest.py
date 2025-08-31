import os
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMB_MODEL = os.getenv("EMBEDDINGS_MODEL") or "text-embedding-3-large"
DATA_DIR = os.getenv("DATA_DIR")
VECTOR_DIR = os.getenv("VECTOR_DIR")
INDEX_NAME = os.getenv("INDEX_NAME")
INDEX_PATH = os.path.join(VECTOR_DIR, INDEX_NAME)

def get_embeddings():
    return OpenAIEmbeddings(model=EMB_MODEL)

def load_txt_docs():
    paths = sorted(glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True))
    print(f"[INFO] Searching for .txt files in: {os.path.abspath(DATA_DIR)}")
    print(f"[INFO] Found {len(paths)} file(s)")
    docs = []
    for p in paths:
        print(f"[LOAD] Reading {p} ...")
        try:
            file_docs = TextLoader(p, autodetect_encoding=True).load()
            print(f"[OK] Loaded {len(file_docs)} document(s) from {p}")
            docs.extend(file_docs)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return docs

def build_index():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    docs = load_txt_docs()
    if not docs:
        print("[ERROR] No documents found. Exiting.")
        return
    print(f"[INFO] Splitting {len(docs)} document(s) into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Created {len(chunks)} chunk(s) total.")
    print(f"[INFO] Using embeddings provider='openai' model='{EMB_MODEL}'")
    embeddings = get_embeddings()
    print("[INFO] Building FAISS index...")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_PATH)
    print(f"[DONE] Index saved at: {INDEX_PATH}")

if __name__ == "__main__":
    build_index()
    print("OK")
