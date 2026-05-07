# Graph Report - .  (2026-05-07)

## Corpus Check
- 11 files · ~6,500 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 106 nodes · 152 edges · 13 communities detected
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_RAG Retrieval Pipeline|RAG Retrieval Pipeline]]
- [[_COMMUNITY_Dual Backend Config|Dual Backend Config]]
- [[_COMMUNITY_Local RAG (ollama)|Local RAG (ollama)]]
- [[_COMMUNITY_Structured QA Backends|Structured QA Backends]]
- [[_COMMUNITY_Agentic Loop (5-iter)|Agentic Loop (5-iter)]]
- [[_COMMUNITY_Agentic Loop Demo|Agentic Loop Demo]]
- [[_COMMUNITY_minimal_rag Functions|minimal_rag Functions]]
- [[_COMMUNITY_RAG Concepts & Models|RAG Concepts & Models]]
- [[_COMMUNITY_Async Cached Q&A|Async Cached Q&A]]
- [[_COMMUNITY_Async Q&A Demo|Async Q&A Demo]]
- [[_COMMUNITY_Async & Cache Patterns|Async & Cache Patterns]]
- [[_COMMUNITY_Agentic Loop Pattern|Agentic Loop Pattern]]
- [[_COMMUNITY_Chunking Utility|Chunking Utility]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 8 edges
2. `execute_tool()` - 8 edges
3. `main()` - 8 edges
4. `main()` - 8 edges
5. `answer()` - 7 edges
6. `call_with_backoff` - 7 edges
7. `execute_tool()` - 6 edges
8. `call_ollama` - 6 edges
9. `main (structured_qa)` - 5 edges
10. `hybrid_retrieve()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `StructJSON` --references--> `README (claude_qa_llm)`  [EXTRACTED]
  structured_qa.py → README.md
- `call_with_backoff` --references--> `README (claude_qa_llm)`  [EXTRACTED]
  structured_qa.py → README.md
- `GEN_MODEL = qwen3.5:0.8b` --references--> `README (claude_qa_llm)`  [EXTRACTED]
  minimal_rag.py → README.md
- `StructJSON` --conceptually_related_to--> `GEN_MODEL = qwen3.5:0.8b`  [INFERRED]
  structured_qa.py → minimal_rag.py
- `call_ollama` --shares_data_with--> `GEN_MODEL = qwen3.5:0.8b`  [EXTRACTED]
  structured_qa.py → minimal_rag.py

## Hyperedges (group relationships)
- **StructJSON shared contract across Claude and ollama backends** — structured_qa_structjson, structured_qa_call_with_backoff, structured_qa_call_ollama [EXTRACTED 1.00]
- **minimal_rag ingestion and retrieval pipeline** — minimal_rag_chunk_text, minimal_rag_add_docs, minimal_rag_build_index, minimal_rag_retrieve, minimal_rag_answer [EXTRACTED 0.95]
- **qwen3.5:0.8b used by both structured_qa --ollama and minimal_rag** — structured_qa_call_ollama, minimal_rag_answer, minimal_rag_gen_model [EXTRACTED 1.00]

## Communities (14 total, 3 thin omitted)

### Community 0 - "RAG Retrieval Pipeline"
Cohesion: 0.17
Nodes (19): add_docs(), answer(), build_index(), call_with_backoff(), chunk_text(), embed(), hybrid_retrieve(), init_db() (+11 more)

### Community 1 - "Dual Backend Config"
Cohesion: 0.24
Nodes (13): GEN_MODEL = qwen3.5:0.8b, README (claude_qa_llm), build_system_claude, build_system_ollama, cache_control ephemeral, call_ollama, call_with_backoff, exponential backoff with jitter (+5 more)

### Community 2 - "Local RAG (ollama)"
Cohesion: 0.33
Nodes (11): add_docs(), answer(), build_index(), chunk_text(), embed(), init_db(), load_docs(), main() (+3 more)

### Community 3 - "Structured QA Backends"
Cohesion: 0.31
Nodes (10): BaseModel, build_system_claude(), build_system_ollama(), call_ollama(), call_with_backoff(), load_document(), main(), parse_args() (+2 more)

### Community 4 - "Agentic Loop (5-iter)"
Cohesion: 0.33
Nodes (9): calculate_average(), compare_cities(), execute_tool(), Execute a tool and return the result., Run the agentic loop until done or max iterations., run_agent(), search_elevation(), search_population() (+1 more)

### Community 5 - "Agentic Loop Demo"
Cohesion: 0.39
Nodes (7): calculate(), execute_tool(), Run the agentic loop until the agent says it's done or max iterations., Execute a tool and return the result., read_file(), run_agent(), search()

### Community 6 - "minimal_rag Functions"
Cohesion: 0.32
Nodes (8): add_docs, answer (minimal_rag), build_index, embed, EMBED_MODEL = nomic-embed-text, hybrid retrieval (FAISS + BM25), main (minimal_rag), retrieve

### Community 7 - "RAG Concepts & Models"
Cohesion: 0.47
Nodes (4): Cross-Encoder Reranking, Hybrid Retrieval Pattern (Vector + BM25), nomic-embed-text Embedding Model, SQLite + FAISS Vector Store

### Community 8 - "Async Cached Q&A"
Cohesion: 0.7
Nodes (4): ask(), build_system(), main(), print_usage()

### Community 9 - "Async Q&A Demo"
Cohesion: 0.67
Nodes (3): ask(), main(), Send one question and return (question, answer).

## Knowledge Gaps
- **20 isolated node(s):** `Send one question and return (question, answer).`, `StructJSON`, `Load all markdown files from a root directory recursively.     Returns list of t`, `Chunk markdown by heading → paragraph boundaries.      Each chunk is prefixed wi`, `Score each (query, chunk) pair jointly and return top_n by score.      Cross-enc` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GEN_MODEL = qwen3.5:0.8b` connect `Dual Backend Config` to `minimal_rag Functions`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Why does `answer (minimal_rag)` connect `minimal_rag Functions` to `Dual Backend Config`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **What connects `Send one question and return (question, answer).`, `StructJSON`, `Load all markdown files from a root directory recursively.     Returns list of t` to the rest of the system?**
  _20 weakly-connected nodes found - possible documentation gaps or missing edges._