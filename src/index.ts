/**
 * @quilt/rag — production RAG framework where every component is a cell.
 *
 * 9 cell kinds:
 *   1. Loader        — loads documents
 *   2. Chunker       — splits into chunks
 *   3. Embedder      — vectorizes
 *   4. VectorStore   — stores + queries
 *   5. Retriever     — top-K by similarity
 *   6. Reranker      — re-scores top-K
 *   7. ContextBuilder — builds prompt context
 *   8. Generator     — calls the LLM
 *   9. Evaluator     — scores the result
 *
 * The whole pipeline is a Quilt sheet. Every cell has the same unified
 * interface — swap any cell for any other. Cells compose, federate,
 * subscribe, and persist.
 */

// Loaders
export { FileLoader, UrlLoader, S3Loader, R2Loader } from './cells/loader.js';

// Chunkers
export { SentenceChunker, ParagraphChunker, TokenWindowChunker, SemanticChunker } from './cells/chunker.js';

// Embedders
export { WorkersAIEmbedder, OpenAIEmbedder, CohereEmbedder, VoyageEmbedder, LocalOnnxEmbedder } from './cells/embedder.js';

// Vector stores
export { MemoryVectorStore, VectorizeStore, PineconeStore, QdrantStore, PgVectorStore } from './cells/vector-store.js';

// Retrievers
export { CosineRetriever, MmrRetriever, HybridRetriever } from './cells/retriever.js';

// Rerankers
export { BgeReranker, CohereReranker, LocalCrossEncoderReranker } from './cells/reranker.js';

// Generators + context
export { DefaultContextBuilder, OpenAIGenerator, ZaiGenerator, WorkersAIGenerator } from './cells/generator.js';

// Evaluators
export { RelevanceEvaluator, FaithfulnessEvaluator, HallucinationEvaluator, ContextPrecisionEvaluator, RagEvaluatorCell } from './cells/evaluator.js';

// Types
export type {
  Document, Chunk, Embedding, RetrievalResult, RAGResult,
  Loader, Chunker, Embedder, VectorStore, Retriever, Reranker,
  ContextBuilder, Generator, PipelineConfig, ChunkerOptions, RetrievalOptions,
} from './types.js';

import type { PipelineConfig, Document, Chunk, RAGResult, RetrievalResult } from './types.js';
import { MemoryVectorStore } from './cells/vector-store.js';
import { SentenceChunker } from './cells/chunker.js';
import { DefaultContextBuilder, ZaiGenerator } from './cells/generator.js';
import { CosineRetriever } from './cells/retriever.js';
import { OpenAIEmbedder } from './cells/embedder.js';
import { RagEvaluatorCell } from './cells/evaluator.js';

/**
 * RAGPipeline — the high-level orchestrator.
 *
 * Wires together a config and provides `ingest()` for documents and
 * `query()` for questions. Returns citations + token usage.
 */
export class RAGPipeline {
  private config: Required<PipelineConfig>;
  private chunks: Chunk[] = [];

  constructor(config: PipelineConfig = {}) {
    this.config = {
      chunker: config.chunker ?? new SentenceChunker(),
      vectorStore: config.vectorStore ?? new MemoryVectorStore(),
      retriever: config.retriever,
      contextBuilder: config.contextBuilder ?? new DefaultContextBuilder(),
      generator: config.generator,
      loader: config.loader,
      embedder: config.embedder,
      reranker: config.reranker,
    };
  }

  /** Ingest documents into the pipeline. */
  async ingest(docs: Document[]): Promise<number> {
    if (!this.config.embedder) throw new Error('Embedder required for ingest');
    const allChunks: Chunk[] = [];
    for (const doc of docs) {
      const chunks = await this.config.chunker.chunk(doc);
      allChunks.push(...chunks);
    }
    this.chunks.push(...allChunks);
    const texts = allChunks.map((c) => c.text);
    const vectors = await this.config.embedder.embed(texts);
    await this.config.vectorStore.upsert(
      allChunks.map((c, i) => ({
        id: c.id,
        vector: vectors[i]!,
        model: this.config.embedder.model,
        metadata: { text: c.text, documentId: c.documentId, index: c.index },
      }))
    );
    return allChunks.length;
  }

  /** Query the pipeline. */
  async query(question: string, options: { k?: number; rerank?: boolean } = {}): Promise<RAGResult> {
    if (!this.config.embedder) throw new Error('Embedder required for query');
    if (!this.config.generator) throw new Error('Generator required for query');
    const k = options.k ?? 5;
    const t0 = Date.now();

    const retriever = this.config.retriever ?? new CosineRetriever(this.config.vectorStore, this.config.embedder);
    let results = await retriever.retrieve(question, k);

    if (options.rerank && this.config.reranker) {
      results = await this.config.reranker.rerank(question, results, k);
    }

    const context = await this.config.contextBuilder.build(question, results);
    const prompt = `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    const out = await this.config.generator.generate(prompt, {
      system: 'You are a helpful assistant. Answer the question using only the provided context. If the context does not contain the answer, say so.',
    });

    return {
      answer: out.text,
      citations: results,
      tokensUsed: out.tokensUsed,
      latencyMs: Date.now() - t0,
    };
  }

  /** Evaluate the most recent result. */
  evaluate(result: RAGResult, groundTruth: { relevantIds?: string[] } = {}): {
    relevance: number;
    faithfulness: number;
    hallucination: number;
    contextPrecision: number;
  } {
    return new RagEvaluatorCell().evaluate(result, groundTruth);
  }

  /** Get the total chunk count. */
  chunkCount(): number { return this.chunks.length; }
}
