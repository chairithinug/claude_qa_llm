# claude_qa_llm

A CLI toolkit for document Q&A using Claude. Ranges from a simple single-document chatbot to a full RAG pipeline with hybrid retrieval and cross-encoder reranking.

---

## Tools

| File | What it does |
|---|---|
| `structured_QA_RAG.py` | Full RAG pipeline — load a directory of markdown docs, chunk, embed, retrieve, rerank, answer with Claude |
| `structured_qa.py` | Single-document Q&A with prompt caching and structured JSON output |
| `minimal_rag.py` | Minimal RAG prototype with FAISS + BM25 hybrid search and local LLM via ollama |
| `qa.py` | Simple streaming Q&A for a single document |
| `embedding.py` | Standalone embedding example using ollama |

---

## structured_QA_RAG.py

The main tool. Ingests a directory of markdown files into a SQLite + FAISS index, then answers questions using a three-stage retrieval pipeline backed by Claude Sonnet.

### Pipeline

```
Markdown files
     │
     ▼
 chunk_text()          Heading-aware chunking: splits on # headings then blank
     │                 lines (paragraphs). Each chunk is prefixed with its
     │                 section heading. Small paragraphs are merged; oversized
     │                 ones fall back to word-splitting with overlap.
     ▼
 add_docs()            Embeds each chunk with nomic-embed-text (via ollama)
     │                 and stores text + embedding + filename in SQLite.
     │                 INSERT OR IGNORE makes re-runs safe.
     ▼
 build_index()         Loads all embeddings from SQLite into a FAISS
     │                 IndexFlatL2 (exact search, no training required).
     │                 Builds a BM25Okapi index over the same corpus.
     ▼
── per query ──────────────────────────────────────────────────────────
     ▼
 hybrid_retrieve()     Fetches k*3=60 candidates from FAISS, scores them
     │                 with BM25, normalizes both, combines (α=0.7 semantic
     │                 + 0.3 lexical), returns top 20.
     ▼
 rerank()              Cross-encoder (ms-marco-MiniLM-L-6-v2) scores each
     │                 (query, chunk) pair jointly — more accurate than
     │                 independent embeddings. Returns top 10.
     ▼
 answer()              Assembles context into the user message (system prompt
                       is a static cached constant). Calls Claude Sonnet with
                       JSON schema output_config. Appends Q+A to history for
                       multi-turn follow-ups.
```

### Setup

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your_key" > .env

# ollama must be running with the embedding model pulled
ollama pull nomic-embed-text
```

### Usage

```bash
# Interactive mode
python structured_QA_RAG.py /path/to/docs

# Single question
python structured_QA_RAG.py /path/to/docs -q "What is the installation process?"

# Tune retrieval hyperparameters
python structured_QA_RAG.py /path/to/docs -k 30 --top-n 15 --alpha 0.5

# Full help
python structured_QA_RAG.py --help
```

On first run the cross-encoder model (~85MB) is downloaded once and cached by `sentence-transformers`.

Delete `rag.db` before re-ingesting with different documents or chunk settings:
```bash
rm rag.db && python structured_QA_RAG.py /path/to/docs
```

### Flags

| Flag | Default | Effect |
|---|---|---|
| `directory` | `.` | Root directory to load markdown files from |
| `-q`, `--question` | — | Ask a single question and exit; omit for interactive mode |
| `-k` | `20` | Candidates fetched by hybrid retrieval before reranking |
| `--top-n` | `10` | Chunks passed to Claude after reranking |
| `--alpha` | `0.7` | Semantic (vector) vs lexical (BM25) weight, `0.0`–`1.0` |

**Tuning tips:**
- Increase `-k` if relevant chunks are being missed; decrease to speed up reranking
- Lower `--alpha` (e.g. `0.5`) for docs with specific terminology, version numbers, or code identifiers where keyword matching matters more
- Increase `--top-n` for broad questions that span multiple sections

### Chunk parameters

These are set in code rather than flags as they require re-ingestion to take effect:

| Parameter | Default | Location | Effect |
|---|---|---|---|
| `max_words` | `150` | `chunk_text()` | Max words per chunk |
| `min_words` | `30` | `chunk_text()` | Min before merging adjacent paragraphs |

### Output

Answers are printed to stdout. Cache stats and the reranked context list are printed to stderr each turn so you can inspect what Claude saw without polluting piped output.

---

## structured_qa.py

Single-document Q&A. Loads one file, caches it in the Claude system prompt for 5 minutes, then answers questions from that cache — no re-tokenizing the document on each turn.

```bash
python structured_qa.py path/to/document.md
```

Responses are structured JSON `{"Q": "...", "A": "..."}`. Conversation history is maintained across turns so follow-up questions work correctly.

---

## minimal_rag.py

Fully local RAG — no Claude API required. Embeds with `nomic-embed-text` and answers with a small local model via ollama. Good for offline use or testing the pipeline without API cost.

### Pipeline

```
Markdown files → chunk_text() → embed() → SQLite + IndexFlatL2 + BM25
                                                        │
                                              per query │
                                                        ▼
                                              hybrid retrieve (vector + BM25)
                                                        │
                                                        ▼
                                              ollama.chat() → answer
```

### Setup

```bash
ollama pull nomic-embed-text
ollama pull qwen3:0.6b
```

### Usage

```bash
# Interactive mode
python minimal_rag.py /path/to/docs

# Single question
python minimal_rag.py /path/to/docs -q "Your question"
```

Delete `rag.db` before re-ingesting with different documents or chunk settings.

### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `size` | 150 | Max words per chunk |
| `overlap` | 30 | Shared words between adjacent chunks |
| `k` | 10 | Chunks passed to the LLM |
| `alpha` | 0.7 | Semantic vs lexical weight in hybrid retrieval |

---

## Stack

| Package | Role |
|---|---|
| `anthropic` | Claude API — structured output, prompt caching, retry |
| `python-dotenv` | Loads `ANTHROPIC_API_KEY` from `.env` |
| `pydantic` | Validates structured JSON responses |
| `ollama` | Local embeddings (`nomic-embed-text`) |
| `faiss-cpu` | Exact vector similarity search (IndexFlatL2) |
| `rank_bm25` | BM25 keyword scoring for hybrid retrieval |
| `sentence-transformers` | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |
| `numpy` | Vector operations |
| `tqdm` | Progress bars during ingestion |

---

## Setup

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your_key" > .env
ollama pull nomic-embed-text
```
