/**
 * @quilt/rag — core types
 *
 * Every component of a RAG pipeline is a cell. Cells have unified interfaces
 * so any cell can be swapped for any other cell of the same kind.
 *
 * The 8 cell kinds:
 *   1. Loader        — loads documents from a source
 *   2. Chunker       — splits documents into chunks
 *   3. Embedder      — vectorizes text
 *   4. VectorStore   — stores and queries vectors
 *   5. Retriever     — top-K by similarity
 *   6. Reranker      — re-scores top-K
 *   7. ContextBuilder — builds the prompt context
 *   8. Generator     — calls the LLM
 *   9. Evaluator     — scores the result
 *
 * The whole pipeline is a Quilt sheet. Each cell has inputs (depends_on)
 * and outputs (cells that depend on it). Reactive propagation means
 * changing a parameter re-runs the downstream cells automatically.
 */

// ──────────────────────────────────────────────────────────────────────────
//  Document
// ──────────────────────────────────────────────────────────────────────────

/** A document loaded from a source. */
export interface Document {
  /** Stable id (e.g. file path, URL). */
  id: string;
  /** Raw text. */
  text: string;
  /** Optional metadata. */
  metadata?: Record<string, unknown>;
}

// ──────────────────────────────────────────────────────────────────────────
//  Chunks
// ──────────────────────────────────────────────────────────────────────────

/** A chunk of a document. */
export interface Chunk {
  /** Stable id (document id + chunk index). */
  id: string;
  /** The chunk text. */
  text: string;
  /** The parent document id. */
  documentId: string;
  /** 0-based index of this chunk in the document. */
  index: number;
  /** Optional metadata. */
  metadata?: Record<string, unknown>;
}

// ──────────────────────────────────────────────────────────────────────────
//  Embeddings
// ──────────────────────────────────────────────────────────────────────────

/** A vector embedding of a chunk. */
export interface Embedding {
  /** Same id as the chunk. */
  id: string;
  /** The vector. */
  vector: number[];
  /** The embedding model id. */
  model: string;
  /** Optional metadata. */
  metadata?: Record<string, unknown>;
}

// ──────────────────────────────────────────────────────────────────────────
//  Retrieval
// ──────────────────────────────────────────────────────────────────────────

/** A retrieved chunk with its score. */
export interface RetrievalResult {
  /** The chunk id. */
  id: string;
  /** The chunk text. */
  text: string;
  /** The similarity score (0-1, higher is better). */
  score: number;
  /** The document id. */
  documentId: string;
  /** Optional metadata. */
  metadata?: Record<string, unknown>;
}

// ──────────────────────────────────────────────────────────────────────────
//  Generation
// ──────────────────────────────────────────────────────────────────────────

/** The result of a RAG query. */
export interface RAGResult {
  /** The generated answer. */
  answer: string;
  /** The citations — which chunks were used. */
  citations: RetrievalResult[];
  /** The total tokens used. */
  tokensUsed?: number;
  /** Latency in ms. */
  latencyMs: number;
}

// ──────────────────────────────────────────────────────────────────────────
//  Cell interfaces
// ──────────────────────────────────────────────────────────────────────────

/** A loader cell. */
export interface Loader {
  load(source: string, options?: Record<string, unknown>): Promise<Document[]>;
}

/** A chunker cell. */
export interface Chunker {
  chunk(doc: Document, options?: Record<string, unknown>): Promise<Chunk[]>;
}

/** An embedder cell. */
export interface Embedder {
  /** Identifier for the model (e.g. "@cf/baai/bge-small-en-v1.5"). */
  readonly model: string;
  /** Dimensionality of the vectors. */
  readonly dimensions: number;
  embed(text: string | string[]): Promise<number[][]>;
}

/** A vector store cell. */
export interface VectorStore {
  upsert(embeddings: Embedding[]): Promise<void>;
  query(vector: number[], k: number, filter?: Record<string, unknown>): Promise<RetrievalResult[]>;
  delete(ids: string[]): Promise<void>;
  /** Count of vectors in the store. */
  count(): Promise<number>;
}

/** A retriever cell. */
export interface Retriever {
  retrieve(query: string, k: number, options?: Record<string, unknown>): Promise<RetrievalResult[]>;
}

/** A reranker cell. */
export interface Reranker {
  rerank(query: string, results: RetrievalResult[], k?: number): Promise<RetrievalResult[]>;
}

/** A context builder cell. */
export interface ContextBuilder {
  build(query: string, results: RetrievalResult[], options?: Record<string, unknown>): Promise<string>;
}

/** A generator cell. */
export interface Generator {
  generate(prompt: string, options?: Record<string, unknown>): Promise<{ text: string; tokensUsed?: number }>;
}

// ──────────────────────────────────────────────────────────────────────────
//  Pipeline
// ──────────────────────────────────────────────────────────────────────────

/** The full pipeline configuration. */
export interface PipelineConfig {
  loader?: Loader;
  chunker?: Chunker;
  embedder?: Embedder;
  vectorStore?: VectorStore;
  retriever?: Retriever;
  reranker?: Reranker;
  contextBuilder?: ContextBuilder;
  generator?: Generator;
}

/** Default chunk options. */
export interface ChunkerOptions {
  strategy?: 'sentence' | 'paragraph' | 'token-window' | 'semantic';
  maxTokens?: number;
  overlap?: number;
}

/** Default retrieval options. */
export interface RetrievalOptions {
  k?: number;
  /** Optional metadata filter. */
  filter?: Record<string, unknown>;
  /** Optional minimum score threshold. */
  minScore?: number;
}
