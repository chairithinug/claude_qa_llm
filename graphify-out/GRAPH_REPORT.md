# Graph Report - .  (2026-05-07)

## Corpus Check
- Corpus is ~6,311 words - fits in a single context window. You may not need a graph.

## Summary
- 93 nodes · 145 edges · 11 communities detected
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_RAG Retrieval Pipeline|RAG Retrieval Pipeline]]
- [[_COMMUNITY_Local RAG (ollama)|Local RAG (ollama)]]
- [[_COMMUNITY_RAG Concepts & Models|RAG Concepts & Models]]
- [[_COMMUNITY_Agentic Loop (5-iter)|Agentic Loop (5-iter)]]
- [[_COMMUNITY_Structured QA (Claude)|Structured QA (Claude)]]
- [[_COMMUNITY_Agentic Loop Demo|Agentic Loop Demo]]
- [[_COMMUNITY_Async Cached Q&A|Async Cached Q&A]]
- [[_COMMUNITY_Streaming Q&A|Streaming Q&A]]
- [[_COMMUNITY_Async & Cache Patterns|Async & Cache Patterns]]
- [[_COMMUNITY_Async Q&A Demo|Async Q&A Demo]]
- [[_COMMUNITY_Agentic Loop Pattern|Agentic Loop Pattern]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 8 edges
2. `main()` - 8 edges
3. `execute_tool()` - 8 edges
4. `answer()` - 7 edges
5. `execute_tool()` - 6 edges
6. `main()` - 5 edges
7. `hybrid_retrieve()` - 4 edges
8. `retrieve()` - 4 edges
9. `main()` - 4 edges
10. `Prompt Caching Pattern` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Cache Warmup Pattern` --conceptually_related_to--> `Prompt Caching Pattern`  [INFERRED]
  async_cached_demo.py → structured_qa.py

## Hyperedges (group relationships)
- **Full RAG Pipeline** — structured_qa_rag, sqlite_vector_store, hybrid_retrieval_pattern, crossencoder_reranking, nomic_embed_text, structjson_model, prompt_caching_pattern [EXTRACTED 1.00]
- **Local RAG Pipeline (no API)** — minimal_rag, sqlite_vector_store, hybrid_retrieval_pattern, nomic_embed_text [EXTRACTED 1.00]
- **Async + Prompt Cache Combination** — async_cached_demo, concurrent_async_pattern, cache_warmup_pattern, prompt_caching_pattern [EXTRACTED 1.00]
- **Agentic Loop Demos** — agentic_loop_demo, agentic_loop_5iter_demo, agentic_loop_pattern [EXTRACTED 1.00]
- **Structured JSON Output Users** — structured_qa, structured_qa_rag, structjson_model [EXTRACTED 1.00]

## Communities (12 total, 1 thin omitted)

### Community 0 - "RAG Retrieval Pipeline"
Cohesion: 0.19
Nodes (18): add_docs(), answer(), build_index(), call_with_backoff(), chunk_text(), embed(), hybrid_retrieve(), init_db() (+10 more)

### Community 1 - "Local RAG (ollama)"
Cohesion: 0.33
Nodes (11): add_docs(), answer(), build_index(), chunk_text(), embed(), init_db(), load_docs(), main() (+3 more)

### Community 2 - "RAG Concepts & Models"
Cohesion: 0.35
Nodes (6): Cross-Encoder Reranking, Exponential Backoff with Jitter, Hybrid Retrieval Pattern (Vector + BM25), nomic-embed-text Embedding Model, SQLite + FAISS Vector Store, StructJSON Pydantic Model

### Community 3 - "Agentic Loop (5-iter)"
Cohesion: 0.33
Nodes (9): calculate_average(), compare_cities(), execute_tool(), Execute a tool and return the result., Run the agentic loop until done or max iterations., run_agent(), search_elevation(), search_population() (+1 more)

### Community 4 - "Structured QA (Claude)"
Cohesion: 0.33
Nodes (8): BaseModel, build_system(), call_with_backoff(), load_document(), main(), print_cache_stats(), StructJSON, StructJSON

### Community 5 - "Agentic Loop Demo"
Cohesion: 0.39
Nodes (7): calculate(), execute_tool(), Run the agentic loop until the agent says it's done or max iterations., Execute a tool and return the result., read_file(), run_agent(), search()

### Community 6 - "Async Cached Q&A"
Cohesion: 0.7
Nodes (4): ask(), build_system(), main(), print_usage()

### Community 7 - "Streaming Q&A"
Cohesion: 0.7
Nodes (4): build_system(), load_document(), main(), print_cache_stats()

### Community 8 - "Async & Cache Patterns"
Cohesion: 0.6
Nodes (3): Cache Warmup Pattern, Concurrent Async Pattern with asyncio.gather, Prompt Caching Pattern

### Community 9 - "Async Q&A Demo"
Cohesion: 0.67
Nodes (3): ask(), main(), Send one question and return (question, answer).

## Knowledge Gaps
- **9 isolated node(s):** `Send one question and return (question, answer).`, `Load all markdown files from a root directory recursively.     Returns list of t`, `Chunk markdown by heading → paragraph boundaries.      Each chunk is prefixed wi`, `Score each (query, chunk) pair jointly and return top_n by score.      Cross-enc`, `Retrieve context via hybrid search and answer using Claude with structured outpu` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StructJSON` connect `Structured QA (Claude)` to `RAG Retrieval Pipeline`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `Prompt Caching Pattern` connect `Async & Cache Patterns` to `RAG Concepts & Models`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **What connects `Send one question and return (question, answer).`, `Load all markdown files from a root directory recursively.     Returns list of t`, `Chunk markdown by heading → paragraph boundaries.      Each chunk is prefixed wi` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._