#!/usr/bin/env python3
"""RAG system using Claude API with structured output and prompt caching."""

import re
import sqlite3
import json
import numpy as np
import faiss
import anthropic
import random
import argparse
import sys
import time
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
DB_PATH = "rag.db"
EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Pydantic model mirrors the output_config JSON schema below — double validation:
# output_config enforces shape at the API level, model_validate_json catches edge cases before use
class StructJSON(BaseModel):
    Q: str = Field(min_length=1)
    A: str = Field(min_length=1)


def load_docs_from_directory(root_dir):
    """Load all markdown files from a root directory recursively.
    Returns list of tuples: (content, filename)
    """
    docs = []
    root_path = Path(root_dir)

    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")

    # sorted() gives deterministic ingestion order across runs, avoiding index drift
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
        description=(
            "RAG system using Claude API over a directory of markdown files.\n"
            "Chunks documents, embeds with nomic-embed-text, retrieves with hybrid\n"
            "vector+BM25 search, reranks with a cross-encoder, and answers with Claude Sonnet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to load markdown files from (default: current directory)"
    )
    parser.add_argument(
        "-q", "--question",
        default=None,
        help="Ask a single question and exit. Omit to enter interactive mode."
    )
    parser.add_argument(
        "-k",
        type=int,
        default=20,
        help="Number of candidates fetched by hybrid retrieval before reranking (default: 20)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        dest="top_n",
        help="Number of chunks passed to Claude after reranking (default: 10)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for semantic (vector) vs lexical (BM25) scores, 0.0–1.0 (default: 0.7)"
    )
    return parser.parse_args()


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

    # Migrate existing DBs that predate the filename column; ALTER TABLE fails if it already exists
    try:
        c.execute("ALTER TABLE documents ADD COLUMN filename TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    return conn


def chunk_text(text, filename, max_words=150, min_words=30):
    """Chunk markdown by heading → paragraph boundaries.

    Each chunk is prefixed with its section heading so retrieved text carries
    its own context (e.g. "## Installation" tells the LLM which section it's from).
    Small paragraphs are merged up to max_words; paragraphs that exceed max_words
    fall back to word-splitting with a 50-word overlap.

    Returns list of tuples: (chunk_text, filename)
    """
    # re.split with a capture group keeps the headings in the result list:
    # [pre_heading_content, heading1, body1, heading2, body2, ...]
    parts = re.split(r'^(#{1,6}\s+.+)$', text, flags=re.MULTILINE)

    # Pair each body with its heading (first part has no heading)
    sections = []
    sections.append(("", parts[0]))
    for i in range(1, len(parts) - 1, 2):
        sections.append((parts[i].strip(), parts[i + 1]))

    chunks = []

    for heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]

        # Merge adjacent small paragraphs so trivially short sections aren't retrieved alone
        buffer = ""
        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(candidate.split()) <= max_words:
                buffer = candidate
            else:
                if buffer:
                    prefix = heading + "\n\n" if heading else ""
                    chunks.append((prefix + buffer, filename))
                # Single paragraph too long — word-split it with overlap
                if len(para.split()) > max_words:
                    words = para.split()
                    for i in range(0, len(words), max_words - 50):
                        sub = " ".join(words[i:i + max_words])
                        prefix = heading + "\n\n" if heading else ""
                        chunks.append((prefix + sub, filename))
                    buffer = ""
                else:
                    buffer = para

        if buffer:
            prefix = heading + "\n\n" if heading else ""
            chunks.append((prefix + buffer, filename))

    return chunks


def embed(text):
    # Lazy import: keeps startup fast and avoids a hard dependency if ollama isn't running
    import ollama
    return ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )["embedding"]


def add_docs(conn, docs):
    c = conn.cursor()
    for doc, filename in tqdm(docs, desc="Adding documents"):
        vec = embed(doc)
        # INSERT OR IGNORE skips chunks already in the DB, making re-runs safe
        c.execute(
            "INSERT OR IGNORE INTO documents (text, embedding, filename) VALUES (?, ?, ?)",
            (doc, json.dumps(vec), filename)
        )
    # Single commit after the loop — per-row commits are ~100x slower on large ingests
    conn.commit()


def build_index(conn):
    c = conn.cursor()
    c.execute("SELECT id, text, embedding FROM documents")
    rows = c.fetchall()

    if not rows:
        raise ValueError("No documents in database. Add documents before building the index.")

    ids, texts, vectors = [], [], []
    for doc_id, text, emb in rows:
        ids.append(doc_id)
        texts.append(text)
        vectors.append(json.loads(emb))

    # FAISS requires float32; Python/numpy defaults to float64
    vectors = np.array(vectors).astype("float32")
    print(f"Built index with {len(vectors)} vectors")
    dim = vectors.shape[1]

    # IndexFlatL2 does exact brute-force search — no training required and correct at any corpus size
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # BM25 is built over the same corpus so hybrid scoring uses aligned indices
    bm25 = BM25Okapi([t.split() for t in texts])

    return index, ids, texts, bm25


def _normalize(scores):
    # Maps scores to [0, 1] so vector (L2 distance) and BM25 (term frequency) are on the same scale
    min_s = min(scores)
    max_s = max(scores)
    # 1e-8 epsilon prevents division by zero when all scores are identical
    return [(s - min_s) / (max_s - min_s + 1e-8) for s in scores]


def hybrid_retrieve(index, ids, texts, bm25, query, k=5, alpha=0.7):
    # Over-fetch candidates so BM25 has room to promote keyword matches that vector search ranked lower
    # Cap at len(ids) to avoid requesting more results than documents exist
    n_candidates = min(k * 3, len(ids))

    q_vec = np.array([embed(query)]).astype("float32")
    D, I = index.search(q_vec, n_candidates)

    candidate_idx = I[0]
    # Convert L2 distance to a similarity score: smaller distance → higher score
    vec_scores = _normalize([1 / (1 + d) for d in D[0]])

    # Score the same candidates with BM25 for keyword-match signal
    bm25_all = bm25.get_scores(query.split())
    bm25_scores = _normalize([bm25_all[i] for i in candidate_idx])

    # alpha weights vector (semantic) vs BM25 (lexical); 0.7 favours semantic by default
    combined = [alpha * v + (1 - alpha) * b for v, b in zip(vec_scores, bm25_scores)]

    ranked = sorted(
        zip([texts[i] for i in candidate_idx], combined),
        key=lambda x: x[1],
        reverse=True,
    )
    return [text for text, _ in ranked[:k]]


def print_cache_stats(usage) -> None:
    created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    uncached = getattr(usage, "input_tokens", 0) or 0
    output = getattr(usage, "output_tokens", 0) or 0

    if created:
        print(f"  [cache write: {created:,} tokens]", file=sys.stderr)
    if read:
        print(f"  [cache hit: {read:,} tokens saved]", file=sys.stderr)
    if not created and not read:
        print(f"  [uncached input: {uncached:,} tokens]", file=sys.stderr)
    if output:
        print(f"  [output: {output:,} tokens]", file=sys.stderr)


def call_with_backoff(client, **kwargs):
    max_retries = 5
    base_delay = 0.5  # seconds; doubles each attempt: 0.5 → 1 → 2 → 4 → 8s

    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)
            text_output = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

            parsed = StructJSON.model_validate_json(text_output)
            return response, parsed

        except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError) as e:
            if attempt == max_retries - 1:
                raise e

            # Jitter prevents thundering-herd if multiple clients hit a rate limit simultaneously
            delay = random.uniform(0, base_delay * (2 ** attempt))
            print(f"  [retry {attempt+1}] waiting {delay:.2f}s...", file=sys.stderr)
            time.sleep(delay)


# Module-level constant so the same bytes are sent every call — required for cache_control to hit
SYSTEM_PROMPT = [
    {
        "type": "text",
        "text": (
            "You are a helpful assistant. Answer questions accurately and "
            "concisely based on the context provided in each user message. "
            "If the answer is not in the context, say: 'I could not find this in the provided context.' Do not guess. "
            "Always respond in JSON format {\"Q\": ..., \"A\": ...}."
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


def rerank(reranker, query, chunks, top_n=10):
    """Score each (query, chunk) pair jointly and return top_n by score.

    Cross-encoders see both texts together, which is more accurate than
    comparing independent embeddings. Scores are raw logits (no fixed range)
    — only their relative order matters here.
    """
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_n]]


def answer(client, reranker, index, ids, texts, bm25, query, history: list, k=20, top_n=10, alpha=0.7):
    """Retrieve context via hybrid search and answer using Claude with structured output.

    history is a list of prior {"role": ..., "content": ...} turns, mutated in place.
    """
    # Hybrid retrieval casts a wide net; cross-encoder reranking picks the most relevant from it
    candidates = hybrid_retrieve(index, ids, texts, bm25, query, k=k, alpha=alpha)
    context_docs = rerank(reranker, query, candidates, top_n=top_n)

    print("\n--- Retrieved context (reranked order) ---", file=sys.stderr)
    for i, chunk in enumerate(context_docs, 1):
        preview = chunk.replace("\n", " ")[:120]
        print(f"  [{i}] {preview}...", file=sys.stderr)
    print(file=sys.stderr)

    context = "\n---\n".join(context_docs)

    # Context goes in the user message so the static system prompt stays cacheable
    history.append({"role": "user", "content": f"<context>\n{context}\n</context>\n\n{query}"})

    try:
        response, parsed = call_with_backoff(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=history,
            # output_config constrains Claude's output to this schema, reducing hallucinated
            # structure and parse failures compared to prompt-only JSON instructions
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "Q": {"type": "string", "minLength": 1},
                            "A": {"type": "string", "minLength": 1},
                        },
                        "required": ["Q", "A"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        if response.usage:
            print_cache_stats(response.usage)

        # Store the full Q+A JSON as the assistant turn so follow-up questions have structured context
        history.append({"role": "assistant", "content": parsed.model_dump_json()})
        return parsed.A

    except (ValueError, anthropic.APIError) as e:
        print(f"\nFailed after retries: {e}", file=sys.stderr)
        # Pop the user turn so history doesn't end on an unanswered message, which would break the API
        history.pop()
        return None


def main():
    args = parse_args()
    docs = load_docs_from_directory(args.directory)

    conn = init_db()
    try:
        all_chunks = []
        for content, filename in tqdm(docs, desc="Processing documents"):
            chunks = chunk_text(content, filename)
            all_chunks.extend(chunks)
        add_docs(conn, all_chunks)
        # After build_index, all data lives in memory (FAISS + BM25); conn is no longer needed
        index, ids, texts, bm25 = build_index(conn)
    finally:
        conn.close()

    client = anthropic.Anthropic()
    # Model is downloaded once on first run and cached locally by sentence-transformers
    reranker = CrossEncoder(RERANK_MODEL)
    history: list = []

    kwargs = dict(k=args.k, top_n=args.top_n, alpha=args.alpha)

    if args.question:
        answer_text = answer(client, reranker, index, ids, texts, bm25, args.question, history, **kwargs)
        if answer_text:
            print(f"\nAnswer: {answer_text}")
    else:
        print(f"\n=== RAG Interactive Mode (k={args.k}, top_n={args.top_n}, alpha={args.alpha}) ===")
        print("Type 'exit' or 'quit' to exit\n")
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

            answer_text = answer(client, reranker, index, ids, texts, bm25, question, history, **kwargs)
            if answer_text:
                print(f"\n> A: {answer_text}\n")


if __name__ == "__main__":
    main()