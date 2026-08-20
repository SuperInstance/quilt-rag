/**
 * Tests for @quilt/rag
 *
 * Uses an in-memory embedder (random vectors) and an in-memory store
 * so tests don't need any API keys.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SentenceChunker, ParagraphChunker, TokenWindowChunker, SemanticChunker,
  MemoryVectorStore, CosineRetriever, MmrRetriever,
  DefaultContextBuilder, RAGPipeline,
  RelevanceEvaluator, FaithfulnessEvaluator, HallucinationEvaluator, ContextPrecisionEvaluator, RagEvaluatorCell,
  type Document, type Chunk, type Embedder, type RetrievalResult,
} from '../src/index.ts';

class FakeEmbedder implements Embedder {
  readonly model = 'fake';
  readonly dimensions = 8;
  // Deterministic vector from text
  async embed(text: string | string[]): Promise<number[][]> {
    const inputs = Array.isArray(text) ? text : [text];
    return inputs.map((t) => {
      const v = new Array(8).fill(0);
      for (let i = 0; i < t.length; i++) {
        v[i % 8]! += t.charCodeAt(i) / 1000;
      }
      const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      return v.map((x) => x / norm);
    });
  }
}

function makeDoc(id: string, text: string): Document {
  return { id, text };
}

test('SentenceChunker — splits on sentence boundaries', async () => {
  const chunker = new SentenceChunker();
  const doc = makeDoc('d1', 'First sentence. Second sentence. Third sentence. ' + 'X. '.repeat(100));
  const chunks = await chunker.chunk(doc, { maxTokens: 50 });
  assert.ok(chunks.length > 0);
  assert.equal(chunks[0]!.documentId, 'd1');
});

test('ParagraphChunker — splits on double newlines', async () => {
  const chunker = new ParagraphChunker();
  const doc = makeDoc('d1', 'Para one.\n\nPara two.\n\nPara three.');
  const chunks = await chunker.chunk(doc);
  assert.ok(chunks.length >= 1);
});

test('TokenWindowChunker — fixed window with overlap', async () => {
  const chunker = new TokenWindowChunker();
  const doc = makeDoc('d1', 'word '.repeat(100));
  const chunks = await chunker.chunk(doc, { maxTokens: 30, overlap: 5 });
  assert.ok(chunks.length >= 2);
});

test('SemanticChunker — chunks at low similarity', async () => {
  const embedder = new FakeEmbedder();
  const chunker = new SemanticChunker(async (t) => (await embedder.embed(t))[0]!, 0.99);
  const doc = makeDoc('d1', 'First. Completely different content here. Another topic.');
  const chunks = await chunker.chunk(doc);
  assert.ok(chunks.length >= 1);
});

test('MemoryVectorStore — basic upsert and query', async () => {
  const store = new MemoryVectorStore();
  await store.upsert([
    { id: 'a', vector: [1, 0, 0, 0, 0, 0, 0, 0], model: 'fake', metadata: { text: 'apple' } },
    { id: 'b', vector: [0, 1, 0, 0, 0, 0, 0, 0], model: 'fake', metadata: { text: 'banana' } },
  ]);
  const results = await store.query([1, 0, 0, 0, 0, 0, 0, 0], 1);
  assert.equal(results[0]!.id, 'a');
});

test('MemoryVectorStore — filter by metadata', async () => {
  const store = new MemoryVectorStore();
  await store.upsert([
    { id: 'a', vector: [1, 0], model: 'fake', metadata: { text: 'apple', documentId: 'doc-1' } },
    { id: 'b', vector: [0.9, 0.1], model: 'fake', metadata: { text: 'apricot', documentId: 'doc-2' } },
  ]);
  const results = await store.query([1, 0], 10, { documentId: 'doc-2' });
  assert.equal(results.length, 1);
  assert.equal(results[0]!.id, 'b');
});

test('MemoryVectorStore — count', async () => {
  const store = new MemoryVectorStore();
  await store.upsert([
    { id: 'a', vector: [1, 0], model: 'fake' },
    { id: 'b', vector: [0, 1], model: 'fake' },
    { id: 'c', vector: [1, 1], model: 'fake' },
  ]);
  assert.equal(await store.count(), 3);
});

test('CosineRetriever — top-K by similarity', async () => {
  const store = new MemoryVectorStore();
  const embedder = new FakeEmbedder();
  // Pre-compute the same vectors the embedder would produce
  const [appleVec] = await embedder.embed('apple');
  const [bananaVec] = await embedder.embed('banana');
  await store.upsert([
    { id: 'a', vector: appleVec, model: 'fake', metadata: { text: 'apple' } },
    { id: 'b', vector: bananaVec, model: 'fake', metadata: { text: 'banana' } },
  ]);
  const retriever = new CosineRetriever(store, embedder);
  const results = await retriever.retrieve('apple', 1);
  assert.equal(results[0]!.id, 'a');
});

test('MmrRetriever — top-K with diversity', async () => {
  const store = new MemoryVectorStore();
  const embedder = new FakeEmbedder();
  const [appleVec] = await embedder.embed('apple');
  const [apricotVec] = await embedder.embed('apricot');
  const [cherryVec] = await embedder.embed('cherry');
  await store.upsert([
    { id: 'a', vector: appleVec, model: 'fake', metadata: { text: 'apple' } },
    { id: 'b', vector: apricotVec, model: 'fake', metadata: { text: 'apricot' } },
    { id: 'c', vector: cherryVec, model: 'fake', metadata: { text: 'cherry' } },
  ]);
  const retriever = new MmrRetriever(store, embedder, 0.5);
  const results = await retriever.retrieve('apple', 2);
  assert.equal(results.length, 2);
});

test('DefaultContextBuilder — respects token budget', async () => {
  const builder = new DefaultContextBuilder();
  const results: RetrievalResult[] = [
    { id: '1', text: 'a'.repeat(100), score: 1, documentId: 'd' },
    { id: '2', text: 'b'.repeat(100), score: 0.9, documentId: 'd' },
  ];
  const ctx = await builder.build('q', results, { maxTokens: 50 });
  assert.ok(ctx.length <= 250);  // 50 tokens * ~4 chars/token + brackets
});

test('RelevanceEvaluator — measures term overlap', () => {
  const ev = new RelevanceEvaluator();
  const result = ev.score('apple banana', [
    { id: '1', text: 'apple is red', score: 0, documentId: 'd' },
    { id: '2', text: 'banana is yellow', score: 0, documentId: 'd' },
  ]);
  assert.ok(result.score > 0);
});

test('FaithfulnessEvaluator — flags unsupported claims', () => {
  const ev = new FaithfulnessEvaluator();
  const result = ev.score('Apples are blue. Bananas are yellow.', [
    { id: '1', text: 'Bananas are yellow', score: 0, documentId: 'd' },
  ]);
  assert.ok(result.unsupported.length > 0);
  assert.ok(result.score < 1);
});

test('HallucinationEvaluator — inverse of faithfulness', () => {
  const ev = new HallucinationEvaluator();
  const r1 = ev.score('apples are blue', [{ id: '1', text: 'no info', score: 0, documentId: 'd' }]);
  const r2 = ev.score('apples are fruits', [{ id: '1', text: 'apples are fruits', score: 0, documentId: 'd' }]);
  assert.ok(r1.score > r2.score);
});

test('ContextPrecisionEvaluator — rewards correct ranking', () => {
  const ev = new ContextPrecisionEvaluator();
  const r = ev.score(['1', '3'], [
    { id: '1', text: '', score: 0, documentId: 'd' },
    { id: '2', text: '', score: 0, documentId: 'd' },
    { id: '3', text: '', score: 0, documentId: 'd' },
  ]);
  assert.ok(r.score > 0);
});

test('RagEvaluatorCell — combined metrics', () => {
  const cell = new RagEvaluatorCell();
  const result = {
    answer: 'Bananas are yellow.',
    citations: [
      { id: '1', text: 'Bananas are yellow', score: 0.9, documentId: 'd' },
    ],
    latencyMs: 100,
  };
  const m = cell.evaluate(result, { relevantIds: ['1'] });
  assert.equal(typeof m.relevance, 'number');
  assert.equal(typeof m.faithfulness, 'number');
  assert.equal(typeof m.hallucination, 'number');
  assert.equal(typeof m.contextPrecision, 'number');
});

test('RAGPipeline — full pipeline with mock embedder', async () => {
  const embedder = new FakeEmbedder();
  const pipeline = new RAGPipeline({
    embedder,
    chunker: new SentenceChunker(),
    vectorStore: new MemoryVectorStore(),
    contextBuilder: new DefaultContextBuilder(),
    generator: {
      async generate(prompt: string) {
        return { text: `Mock answer to: ${prompt.slice(0, 50)}`, tokensUsed: 10 };
      },
    },
  });
  const count = await pipeline.ingest([
    makeDoc('d1', 'The capital of France is Paris. The capital of Japan is Tokyo. The capital of Brazil is Brasilia.'),
  ]);
  assert.ok(count >= 1);

  const result = await pipeline.query('What is the capital of France?');
  assert.ok(result.answer);
  assert.ok(result.citations.length > 0);
  assert.equal(typeof result.latencyMs, 'number');
});

test('RAGPipeline — chunkCount tracks ingested chunks', async () => {
  const embedder = new FakeEmbedder();
  const pipeline = new RAGPipeline({
    embedder,
    chunker: new SentenceChunker(),
  });
  await pipeline.ingest([makeDoc('d1', 'One. Two. Three.')]);
  assert.ok(pipeline.chunkCount() >= 1);
});
