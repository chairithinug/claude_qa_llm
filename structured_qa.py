#!/usr/bin/env python3
"""Document Q&A CLI — loads a file once (prompt-cached), then answers questions about it."""

import sys
import time
import random
import anthropic
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
SOFT_SIZE_LIMIT = 200_000  # characters

# Pydantic model mirrors the output_config schema below.
# Double validation: output_config enforces the shape at the API level,
# model_validate_json catches any edge cases before we use the data.
class StructJSON(BaseModel):
    Q: str = Field(min_length=1)
    A: str = Field(min_length=1)

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
                "concisely based on the document below. "
                "If the answer is not explicitly in the document, say: 'I could not find this in the document.' Do not guess. "
                "Always respond in JSON format {\"Q\": ..., \"A\": ...}.\n\n"
                f"<document>\n{content}\n</document>"
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]


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
    base_delay = 0.5  # seconds; doubles each attempt (0.5 → 1 → 2 → 4 → 8s max)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)
            # Concatenate all text blocks — output_config guarantees a single text block,
            # but this is defensive against future content block changes.
            text_output = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

            parsed = StructJSON.model_validate_json(text_output)

            return response, parsed

        except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError) as e:
            if attempt == max_retries - 1:
                raise e

            # Jitter (random.uniform) prevents thundering-herd if multiple
            # clients retry at the same time after a rate-limit window.
            delay = random.uniform(0, base_delay * (2 ** attempt))
            print(f"  [retry {attempt+1}] waiting {delay:.2f}s...", file=sys.stderr)
            time.sleep(delay)

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

        usage = None

        try:
            response, parsed = call_with_backoff(
                client,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0,
                system=system,
                messages=messages,
                # output_config tells the API to constrain its output to this
                # JSON schema, reducing hallucinated structure and parse failures.
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

            usage = response.usage
            print(f"  A: {parsed.A}")

        except (ValueError, anthropic.APIError) as e:
            print(f"\nFailed after retries: {e}", file=sys.stderr)
            messages.pop()
            continue

        print()  # newline after streamed answer
        if usage:
            print_cache_stats(usage)
        print()

        # Store the full Q+A JSON as the assistant turn so multi-turn follow-ups
        # have the structured context, not free-form text.
        messages.append({
            "role": "assistant",
            "content": parsed.model_dump_json()
        })


if __name__ == "__main__":
    main()
