import sys
from rag.ingest import build_index

def main() -> None:
    build_index()
    print("RAG index built successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)
