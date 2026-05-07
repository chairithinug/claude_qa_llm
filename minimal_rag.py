import sqlite3
import json
import argparse
import numpy as np
import faiss
import ollama
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

DB_PATH = "rag.db"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3:0.6b"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal RAG over a directory of markdown files"
    )
    parser.add_argument(
        "directory", nargs="?", default=".",
        help="Root directory to load markdown files from (default: current directory)"
    )
    parser.add_argument(
        "-q", "--question", default=None,
        help="Ask a single question and exit. Omit to enter interactive mode."
    )
    return parser.parse_args()


def load_docs(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")
    docs = []
    for path in sorted(root.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append((text, str(path)))
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")
    if not docs:
        raise ValueError(f"No markdown files found in {root_dir}")
    return docs


def chunk_text(text, filename, size=150, overlap=30):
    words = text.split()
    # Step by (size - overlap) so adjacent chunks share context across boundaries
    return [
        (" ".join(words[i:i + size]), filename)
        for i in range(0, len(words), size - overlap)
    ]


def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            text TEXT UNIQUE,
            embedding TEXT,
            filename TEXT
        )
    """)
    # Migrate existing DBs that predate the filename column
    try:
        c.execute("ALTER TABLE documents ADD COLUMN filename TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn


def add_docs(conn, chunks):
    c = conn.cursor()
    for text, filename in tqdm(chunks, desc="Embedding"):
        vec = embed(text)
        # INSERT OR IGNORE makes re-runs safe without clearing the DB
        c.execute(
            "INSERT OR IGNORE INTO documents (text, embedding, filename) VALUES (?, ?, ?)",
            (text, json.dumps(vec), filename)
        )
    # Single commit after the loop — per-row commits are much slower
    conn.commit()


def build_index(conn):
    c = conn.cursor()
    c.execute("SELECT id, text, embedding FROM documents")
    rows = c.fetchall()
    if not rows:
        raise ValueError("No documents in database.")

    ids, texts, vectors = [], [], []
    for doc_id, text, emb in rows:
        ids.append(doc_id)
        texts.append(text)
        vectors.append(json.loads(emb))

    # FAISS requires float32; numpy defaults to float64
    vectors = np.array(vectors).astype("float32")
    print(f"Index: {len(vectors)} chunks")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # BM25 built over the same corpus so indices stay aligned with FAISS
    bm25 = BM25Okapi([t.split() for t in texts])
    return index, ids, texts, bm25


def _normalize(scores):
    lo, hi = min(scores), max(scores)
    return [(s - lo) / (hi - lo + 1e-8) for s in scores]


def retrieve(index, ids, texts, bm25, query, k=10, alpha=0.7):
    # Over-fetch so BM25 can promote keyword matches that vector search ranked lower
    n = min(k * 3, len(ids))
    q_vec = np.array([embed(query)]).astype("float32")
    D, I = index.search(q_vec, n)

    # Convert L2 distance to similarity: smaller distance → higher score
    vec_scores = _normalize([1 / (1 + d) for d in D[0]])
    bm25_all = bm25.get_scores(query.split())
    bm25_scores = _normalize([bm25_all[i] for i in I[0]])

    # alpha weights semantic (vector) vs lexical (BM25)
    combined = [alpha * v + (1 - alpha) * b for v, b in zip(vec_scores, bm25_scores)]
    ranked = sorted(zip([texts[i] for i in I[0]], combined), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked[:k]]


def answer(index, ids, texts, bm25, query):
    context = "\n---\n".join(retrieve(index, ids, texts, bm25, query))
    res = ollama.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": f"Answer ONLY using this context:\n{context}\n\nQuestion: {query}"}]
    )
    return res["message"]["content"]


def main():
    args = parse_args()
    docs = load_docs(args.directory)

    conn = init_db()
    try:
        chunks = []
        for content, filename in tqdm(docs, desc="Chunking"):
            chunks.extend(chunk_text(content, filename))
        add_docs(conn, chunks)
        index, ids, texts, bm25 = build_index(conn)
    finally:
        conn.close()

    if args.question:
        print(answer(index, ids, texts, bm25, args.question))
    else:
        print("\n=== RAG Interactive Mode (type 'quit' to exit) ===\n")
        while True:
            try:
                question = input("> Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            print(f"\n> A: {answer(index, ids, texts, bm25, question)}\n")


if __name__ == "__main__":
    main()
