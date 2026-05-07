import sqlite3
import json
import numpy as np
import faiss
import ollama
import random
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def load_docs_from_directory(root_dir):
    """Load all markdown files from a root directory recursively.
    Returns list of tuples: (content, filename)
    """
    docs = []
    root_path = Path(root_dir)

    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")

    md_files = sorted(root_path.rglob("*.md"))

    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    docs.append((content, str(md_file)))
        except Exception as e:
            print(f"Warning: Could not read {md_file}: {e}")

    if not docs:
        raise ValueError(f"No markdown files found in {root_dir}")

    return docs


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG system that loads markdown documents from a directory"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to load markdown files from (default: current directory)"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help="Question to ask the RAG system. If not provided, enters interactive mode"
    )
    return parser.parse_args()


args = parse_args()
docs = load_docs_from_directory(args.directory)

doc_contents = [content for content, _ in docs]
tokenized_docs = [doc.split() for doc in doc_contents]
bm25 = BM25Okapi(tokenized_docs)

def bm25_search(query, k=3):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_k = np.argsort(scores)[-k:][::-1]
    return [doc_contents[i] for i in top_k]


DB_PATH = "rag.db"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3.5:0.8b"

def embed(text):
    return ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )["embedding"]

# def cosine(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# docs = [
#     "RAG stands for Retrieval Augmented Generation.",
#     "It combines vector search with language models.",
#     "Cats are animals."
# ]

# vectors = [embed(d) for d in docs]

# def retrieve(query, k=2):
#     q_vec = embed(query)
#     scores = [cosine(q_vec, v) for v in vectors]
#     top_k = np.argsort(scores)[-k:][::-1]
#     return [docs[i] for i in top_k]

# def answer(query):
#     context = "\n".join(retrieve(query))
#     res = ollama.chat(
#         model=GEN_MODEL,
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"""
#                     Answer using ONLY this context:
#                     {context}
#                     Question: {query}
#                     """
#             }
#         ]
#     )
#     return res["message"]["content"]

# print(answer("What is RAG?"))





def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # c.execute("DELETE FROM documents")
    # conn.commit()
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            text TEXT UNIQUE,
            embedding TEXT,
            filename TEXT
        )
    """)

    # Add filename column if it doesn't exist
    try:
        c.execute("ALTER TABLE documents ADD COLUMN filename TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    return conn

def chunk_text(text, filename, size=300, overlap=50):
    """Chunk text and attach filename metadata.
    Returns list of tuples: (chunk_text, filename)
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = words[i:i + size]
        chunks.append((" ".join(chunk), filename))
    return chunks

def add_docs(conn, docs):
    c = conn.cursor()
    for doc, filename in tqdm(docs, desc="Adding documents"):
        vec = embed(doc)
        c.execute(
            "INSERT OR IGNORE INTO documents (text, embedding, filename) VALUES (?, ?, ?)",
            (doc, json.dumps(vec), filename)
        )
    conn.commit()

def build_index(conn, nlist=100):
    c = conn.cursor()
    c.execute("SELECT id, embedding FROM documents")
    rows = c.fetchall()
    vectors = []
    ids = []
    for row in rows:
        ids.append(row[0])
        vectors.append(json.loads(row[1]))
    vectors = np.array(vectors).astype("float32")
    print(vectors.shape)
    dim = vectors.shape[1]

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    index.train(vectors)
    index.add(vectors)
    return index, ids


def retrieve(conn, index, ids, query, k=5):
    q_vec = np.array([embed(query)]).astype("float32")
    index.nprobe = 10  # how many clusters to check
    D, I = index.search(q_vec, k)
    c = conn.cursor()
    results = []
    print(D, I)
    for idx in I[0]:
        doc_id = ids[idx]
        c.execute("SELECT text FROM documents WHERE id=?", (doc_id,))
        results.append(c.fetchone()[0])
    return results

def normalize(scores):
    min_s = min(scores)
    max_s = max(scores)
    return [(s - min_s) / (max_s - min_s + 1e-8) for s in scores]

def hybrid_search(query, k=5, alpha=0.7):
    # --- vector search ---
    q_vec = np.array([embed(query)]).astype("float32")
    D, I = index.search(q_vec, k)

    vec_scores = [1 / (1 + d) for d in D[0]]
    vec_docs = [doc_contents[i] for i in I[0]]

    # --- bm25 search ---
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # pick same docs for alignment
    bm25_subset = [bm25_scores[i] for i in I[0]]

    # --- normalize ---
    vec_norm = normalize(vec_scores)
    bm25_norm = normalize(bm25_subset)

    # --- combine ---
    final_scores = [
        alpha * v + (1 - alpha) * b
        for v, b in zip(vec_norm, bm25_norm)
    ]

    # sort
    ranked = sorted(
        zip(vec_docs, final_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:k]]


def answer(conn, index, ids, query):
    context = "\n".join(retrieve(conn, index, ids, query))
    print(context)
    res = ollama.chat(
        model=GEN_MODEL,
        messages=[{
            "role": "user",
            "content": f"""
                Answer ONLY using this context:
                {context}
                Question: {query}
            """
        }]
    )
    return res["message"]["content"]

if __name__ == "__main__":
    conn = init_db()
    all_chunks = []
    for content, filename in tqdm(docs, desc="Processing documents"):
        chunks = chunk_text(content, filename)
        all_chunks.extend(chunks)
    add_docs(conn, all_chunks)
    index, ids = build_index(conn)

    if args.question:
        print(answer(conn, index, ids, args.question))
    else:
        print("\n=== RAG Interactive Mode ===")
        print("Type 'exit' or 'quit' to exit\n")
        while True:
            question = input("Enter your question: ").strip()
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if question:
                response = answer(conn, index, ids, question)
                print(f"\nAnswer: {response}\n")