# claude_qa_llm

A CLI tool that lets you ask unlimited questions about any text document using Claude — without re-sending the document on every question.

---

## Why

Querying large documents with an LLM naively re-tokenizes the full document on every request, which is slow and expensive. This tool uses **Anthropic prompt caching** to load the document once, cache it for 5 minutes, and serve all follow-up questions from that cache.

---

## How It Works

1. Load a UTF-8 document from disk
2. Inject it into a system prompt wrapped in `<document>` tags with `cache_control: ephemeral`
3. Open an interactive Q&A loop — each question hits the API with the cached context
4. Stream answers back token-by-token; maintain conversation history for follow-ups
5. Print cache stats (write / hit / uncached tokens) per turn to stderr

---

## Features

- **Prompt caching** — document is tokenized once, reused across all questions (5-min TTL)
- **Streaming output** — answers print in real time as they arrive
- **Multi-turn conversation** — maintains message history so follow-up questions have context
- **Cache stats** — shows tokens written, tokens saved on cache hits, and uncached input per query
- **Size warning** — alerts when document exceeds 200k characters (cache coverage may be partial)
- **Graceful error handling** — API failures discard the failed turn and let you retry

---

## Tools & Stack

| Tool | Role |
|---|---|
| `anthropic` SDK | Claude API client with streaming and prompt caching |
| `python-dotenv` | Loads `ANTHROPIC_API_KEY` from `.env` |
| `claude-sonnet-4-6` | Model used for answering |

---

## Usage

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your_key" > .env
python qa.py path/to/document.txt
```

Then type questions at the `> Q:` prompt. `Ctrl+C` or `Ctrl+D` to exit.
