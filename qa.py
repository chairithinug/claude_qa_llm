#!/usr/bin/env python3
"""Document Q&A CLI — loads a file once (prompt-cached), then answers questions about it."""

import sys
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
SOFT_SIZE_LIMIT = 200_000  # characters


def load_document(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {path}")
    except UnicodeDecodeError:
        sys.exit(f"Error: could not read {path} as UTF-8 text")

    if len(content) > SOFT_SIZE_LIMIT:
        print(
            f"Warning: document is {len(content):,} chars "
            f"(>{SOFT_SIZE_LIMIT:,}). Caching may not cover the full text.",
            file=sys.stderr,
        )
    return content


def build_system(content: str) -> list[dict]:
    # The cache_control block tells Claude to cache everything up to this point
    # (system prompt + doc) for 5 minutes. Subsequent questions reuse this cache
    # instead of re-tokenizing the entire document each turn.
    return [
        {
            "type": "text",
            "text": (
                "You are a helpful assistant. Answer questions accurately and "
                "concisely based on the document below. If the answer is not "
                "in the document, say so.\n\n"
                f"<document>\n{content}\n</document>"
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def print_cache_stats(usage) -> None:
    created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    uncached = getattr(usage, "input_tokens", 0) or 0

    if created:
        print(f"  [cache write: {created:,} tokens]", file=sys.stderr)
    if read:
        print(f"  [cache hit: {read:,} tokens saved]", file=sys.stderr)
    if not created and not read:
        print(f"  [uncached input: {uncached:,} tokens]", file=sys.stderr)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} <path-to-document>")

    path = sys.argv[1]
    content = load_document(path)
    system = build_system(content)

    client = anthropic.Anthropic()
    messages: list[dict] = []

    print(f"Document loaded: {path} ({len(content):,} chars)")
    print("Ask questions about the document. Press Ctrl+C or Ctrl+D to exit.\n")

    while True:
        try:
            question = input("> Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue

        messages.append({"role": "user", "content": question})

        answer_parts: list[str] = []
        usage = None

        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=messages,
            ) as stream:
                print("  A: ", end="", flush=True)
                for chunk in stream.text_stream:
                    print(chunk, end="", flush=True)
                    answer_parts.append(chunk)

                final = stream.get_final_message()
                usage = final.usage

        except anthropic.APIError as e:
            print(f"\nAPI error: {e}", file=sys.stderr)
            messages.pop()  # discard the failed user turn
            continue

        print()  # newline after streamed answer
        if usage:
            print_cache_stats(usage)
        print()

        answer = "".join(answer_parts)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
