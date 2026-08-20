# ◳ quilt-rag

> **The 17th Quilt repo.** Production RAG where every component is a cell. Loader, chunker, embedder, vector store, retriever, reranker, generator, evaluator — composable, addressable, swappable.

```
 ██████╗ ██╗   ██╗██╗██╗     ████████╗     ██████╗  █████╗  ██████╗ 
██╔═══██╗██║   ██║██║██║     ╚══██╔══╝     ██╔══██╗██╔══██╗██╔════╝ 
██║   ██║██║   ██║██║██║        ██║        ██████╔╝███████║██║  ███╗
██║▄▄ ██║██║   ██║██║██║        ██║        ██╔══██╗██╔══██║██║   ██║
╚██████╔╝╚██████╔╝██║███████╗   ██║        ██║  ██║██║  ██║╚██████╔╝
 ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝   ╚═╝        ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 

 Cells-as-pipeline · 5 stores · 5 embedders · 3 rerankers
```

[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![typescript](https://img.shields.io/badge/TypeScript-strict-blue.svg)](./tsconfig.json)
[![node](https://img.shields.io/badge/node-%3E%3D18-green.svg)](./package.json)
[![version](https://img.shields.io/badge/version-0.1.0-orange.svg)](./package.json)

---

## ✦ What is `quilt-rag`?

I built `@quilt/rag` to stop wrestling with spaghetti code whenever I needed to swap a vector store or an embedder. I chose cells over classes because classes lock you into rigid inheritance chains, whereas cells are isolated, composable blocks that you can mix and match freely. Your entire pipeline becomes a "sheet," connecting eight specific cell kinds: loaders, chunkers, embedders, vector stores, retrievers, rerankers, context builders, and generators. It comes ready for production with support for five vector stores (including Pinecone and Qdrant), five embedders (like OpenAI and local ONNX), and three rerankers. This structure makes debugging easy and upgrades painless.

**Why cells, not classes?** Because RAG is fundamentally a graph: loaders feed chunkers, chunkers feed embedders, embedders feed stores, queries flow back through retrievers and rerankers to a generator. Modeling that as a graph — with reactive propagation — means you can:
- A/B test retrievers in production by swapping one cell
- Track which chunks the generator actually used (each chunk is a cell with provenance)
- Federate across tiers (chunk on Jetson, embed on Cloudflare, store in Vectorize)
- Compose RAG with the rest of your app (an `ai` cell right next to your `api` cell)

## ✦ 8 cell kinds

| Cell | What it does | Implementations |
|---|---|---|
| `loader` | Loads documents from a source | file, URL, S3, R2 |
| `chunker` | Splits documents into chunks | sentence, paragraph, token-window, semantic |
| `embedder` | Vectorizes chunks | Workers AI, OpenAI, Cohere, Voyage, local ONNX |
| `vector_store` | Stores + queries vectors | memory, Vectorize, Pinecone, Qdrant, pgvector |
| `retriever` | Top-K by similarity | cosine, MMR, hybrid BM25+vector, with-reranker |
| `reranker` | Re-scores top-K | Workers AI BGE, Cohere, local cross-encoder |
| `context` | Builds the prompt context | top-K, token-budget, MMR |
| `generator` | Calls the LLM | any provider via @quilt/ai |
| `evaluator` | Scores the result | relevance, faithfulness, hallucination, context-precision |

Every cell has a unified interface. Swap any cell for any other — same sheet.

## ✦ Install

```bash
npm install @quilt/rag
```

## ✦ Quick start

```ts
import { RAGPipeline } from '@quilt/rag';
import { parseSheet } from '@quilt/core';

const pipeline = new RAGPipeline();

const sheet = parseSheet(`
cells:
  docs:
    kind: loader
    source: ./corpus
  chunks:
    kind: chunker
    depends_on: [docs]
    strategy: sentence
    max_tokens: 512
  vectors:
    kind: embedder
    depends_on: [chunks]
    provider: workers-ai
    model: "@cf/baai/bge-small-en-v1.5"
  store:
    kind: vector_store
    depends_on: [vectors]
    backend: vectorize
    index: my-corpus
  retriever:
    kind: retriever
    depends_on: [store]
    strategy: cosine
    k: 10
  context:
    kind: context
    depends_on: [retriever]
    max_tokens: 2048
  answer:
    kind: generator
    depends_on: [context]
    provider: zai
    model: glm-4.5
    template: |
      Context: {{context}}
      Question: {{query}}
      Answer:
`);

await pipeline.load(sheet);

const result = await pipeline.query('What is Quilt?');
console.log(result.answer);
console.log(result.citations);  // which chunks were used
```

## ✦ Why this is different

**1. Cells are addressable.** `quilt://rag/corpus#chunks.42` is a real, subscribable cell. You can:
- Subscribe to a chunk and watch it update when re-embedded
- Pull a specific chunk from a federation
- Trace the entire pipeline through the cell graph

**2. Cells are pluggable.** Want to try Cohere embeddings instead of Workers AI? Change one cell. Want to add a reranker? Add a cell. Want to test top-K=5 vs top-K=20? Change one number.

**3. Cells compose with the rest of Quilt.** An `ai` cell in a sheet can call into the RAG pipeline as a cell. A `listener` can fire when a chunk is added. An `api` cell can ingest a URL into the corpus.

**4. Cells federate.** The loader can run on a Codespace, the embedder on Cloudflare Workers, the store in Vectorize, the retriever on a server. Same cell graph, distributed execution.

**5. Cells are auditable.** Every cell has a value history. Replay the entire pipeline deterministically. Diff the result between two runs.

## ✦ Recipes

| Recipe | File | What it does |
|---|---|---|
| Basic Q&A | `examples/basic-qa.yaml` | 1 doc → chunk → embed → store → retrieve → generate |
| PDF corpus | `examples/pdf-corpus.yaml` | 100 PDFs from R2 → chunk → embed (Workers AI) → Vectorize |
| Hybrid search | `examples/hybrid-search.yaml` | BM25 + vector + reranker |
| With reranker | `examples/with-reranker.yaml` | BGE reranker on top of vector retrieval |
| Multi-tenant | `examples/multi-tenant.yaml` | 3 tenants, separate indexes, shared LLM |
| Eval suite | `examples/evaluation-suite.yaml` | relevance + faithfulness + hallucination metrics |

## ✦ Vector store comparison

| Store | When to use | Persistence | Federation |
|---|---|---|---|
| `memory` | Tests, ephemeral | None | Local |
| `vectorize` | Cloudflare-native, global | R2-backed | Yes (Workers) |
| `pinecone` | Managed, scale | Pinecone cloud | Yes (HTTP) |
| `qdrant` | Self-hosted, fast | Disk or memory | Yes (gRPC) |
| `pgvector` | Already using Postgres | Postgres | Yes (SQL) |

All expose the same `VectorStore` interface. Same code, different backend.

## ✦ Embedder comparison

| Embedder | Model | Dimensions | Cost | Speed |
|---|---|---|---|---|
| `workers-ai` | `@cf/baai/bge-small-en-v1.5` | 384 | Free | Fast |
| `workers-ai-large` | `@cf/baai/bge-large-en-v1.5` | 1024 | Free | Medium |
| `openai-small` | `text-embedding-3-small` | 1536 | $0.02/1M tokens | Fast |
| `openai-large` | `text-embedding-3-large` | 3072 | $0.13/1M tokens | Medium |
| `cohere-v3` | `embed-english-v3.0` | 1024 | $0.10/1M tokens | Fast |
| `voyage-3` | `voyage-3` | 1024 | $0.06/1M tokens | Fast |
| `local-bge` | Transformers.js ONNX | 384-1024 | Free | Slow (one-time setup) |

## ✦ The Quilt ecosystem

`quilt-rag` is one of 18 Quilt repos. Each one is a different surface for the same cell model.

| Need | Repo |
|---|---|
| Canonical TS core | [`quilt`](https://github.com/SuperInstance/quilt) |
| Rust binary | [`quilt-rust`](https://github.com/SuperInstance/quilt-rust) |
| Single HTML file | [`quilt-live`](https://github.com/SuperInstance/quilt-live) |
| Cloudflare Workers | [`quilt-cloudflare`](https://github.com/SuperInstance/quilt-cloudflare) |
| Edge ML (Jetson) | [`quilt-jetson`](https://github.com/SuperInstance/quilt-jetson) |
| ESP32 (no_std) | [`quilt-esp32`](https://github.com/SuperInstance/quilt-esp32) |
| Codespace runtime | [`quilt-codespace`](https://github.com/SuperInstance/quilt-codespace) |
| AI cells (4 providers) | [`quilt-ai`](https://github.com/SuperInstance/quilt-ai) |
| Self-improvement loops | [`quilt-evolve`](https://github.com/SuperInstance/quilt-evolve) |
| Multi-device fleet | [`quilt-fleet`](https://github.com/SuperInstance/quilt-fleet) |
| RAG pipeline | **`quilt-rag`** ← you are here |
| 8 SDK primitives | [`@quilt/sdk`](https://github.com/SuperInstance/quilt) |
| Mesh / P2P | [`quilt-mesh`](https://github.com/SuperInstance/quilt-mesh) |
| Agent harness | [`quilt-agent`](https://github.com/SuperInstance/quilt-agent) |
| Time-series cells | [`quilt-time`](https://github.com/SuperInstance/quilt-time) |
| Secrets cells | [`quilt-vault`](https://github.com/SuperInstance/quilt-vault) |
| Computer vision cells | [`quilt-vision`](https://github.com/SuperInstance/quilt-vision) |
| ZK verification | [`quilt-zk`](https://github.com/SuperInstance/quilt-zk) |
| Workflow DAG | [`quilt-flow`](https://github.com/SuperInstance/quilt-flow) |

## ✦ License

Apache 2.0. See [LICENSE](./LICENSE).
