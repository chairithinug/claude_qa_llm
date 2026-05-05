# claude_qa_llm

A CLI toolkit for asking unlimited questions about any text document using Claude — without re-sending the document on every question.

---

## Why

Querying large documents with an LLM naively re-tokenizes the full document on every request, which is slow and expensive. This toolkit uses **Anthropic prompt caching** to load the document once, cache it for 5 minutes, and serve all follow-up questions from that cache.

---

## How It Works

1. Load a UTF-8 document from disk
2. Inject it into a system prompt wrapped in `<document>` tags with `cache_control: ephemeral`
3. Open an interactive Q&A loop — each question hits the API with the cached context
4. Stream answers back and/or return structured output; maintain conversation history for follow-ups
5. Print cache stats (write / hit / uncached tokens) per turn to stderr

---

## Tools

### qa.py (Streaming Q&A)
**Original tool.** Interactive streaming Q&A with real-time output and prompt caching.

**Features:**
- Streaming text output (real-time token delivery)
- Multi-turn conversation with message history
- Cache stats per query (write/hit/uncached tokens)
- Graceful error handling with retry
- Sonnet 4-5 model

### structured_qa.py (Structured JSON Output) ⭐ Recommended
**Modern variant.** Interactive Q&A returning structured JSON with exponential backoff retry logic.

**Features:**
- Structured JSON output (`{"Q": "...", "A": "..."}`)
- JSON schema validation on responses
- Exponential backoff retry on API errors (up to 5 attempts)
- Same prompt caching and multi-turn support
- Sonnet 4-6 model (newer)
- Better for programmatic processing

---

## Examples

### embedding.py
Simple example using local embeddings via [ollama](https://ollama.ai). Tests embedding generation with the `nomic-embed-text` model.

### minimal_rag.py (Work in Progress)
Example RAG system demonstrating:
- Vector similarity search with FAISS
- BM25 keyword search for hybrid ranking
- SQLite document storage
- Chunking and embeddings pipeline

See code comments for usage patterns.

---

## Stack

| Component | Purpose |
|---|---|
| `anthropic` SDK (v0.40.0+) | Claude API with streaming and prompt caching |
| `python-dotenv` | Load `ANTHROPIC_API_KEY` from `.env` |
| `pydantic` | Structured data validation (structured_qa.py) |
| `ollama` | Local embeddings & inference (examples) |
| `numpy` | Vector operations (examples) |
| `faiss-cpu` | Vector similarity search (examples) |
| `rank_bm25` | BM25 ranking (examples) |

---

## Setup

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your_key" > .env
```

---

## Usage

### Streaming Q&A (qa.py)
```bash
python qa.py path/to/document.txt
```

Then type questions at the `> Q:` prompt. Press Ctrl+C or Ctrl+D to exit.

### Structured Q&A (structured_qa.py)
```bash
python structured_qa.py path/to/document.txt
```

Same interactive loop, but responses return as structured JSON with validation.
