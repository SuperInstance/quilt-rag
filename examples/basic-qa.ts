/**
 * Basic Q&A example — load documents, embed, store, retrieve, generate.
 *
 * Usage:
 *   OPENAI_API_KEY=sk-... npx tsx examples/basic-qa.ts
 *
 * Or with z.ai:
 *   ZAI_API_KEY=... npx tsx examples/basic-qa.ts
 */
import { RAGPipeline } from '../src/index.js';

async function main() {
  const zaiKey = process.env.ZAI_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;

  if (!zaiKey && !openaiKey) {
    console.error('Set ZAI_API_KEY or OPENAI_API_KEY');
    process.exit(1);
  }

  const pipeline = new RAGPipeline();

  // Configure providers dynamically based on which key is available
  if (zaiKey) {
    const { ZaiGenerator } = await import('../src/index.js');
    const { ZaiEmbedder } = await import('../src/cells/embedder.js');
    (pipeline as any).config.generator = new ZaiGenerator(zaiKey);
    (pipeline as any).config.embedder = new (ZaiEmbedder as any)(zaiKey);
  } else if (openaiKey) {
    const { OpenAIGenerator, OpenAIEmbedder } = await import('../src/index.js');
    (pipeline as any).config.generator = new OpenAIGenerator(openaiKey!);
    (pipeline as any).config.embedder = new OpenAIEmbedder(openaiKey!);
  }

  // Ingest a small corpus
  const docs = [
    {
      id: 'quilt-intro',
      text: 'Quilt is a reactive, typed, cellular runtime. A spreadsheet where every cell is a live, addressable capability. The grid is the runtime. The cell is the universal IO primitive.',
    },
    {
      id: 'quilt-cells',
      text: 'There are 9 cell kinds in Quilt: value, formula, api, program, sensor, listener, router, io, and ai. Each cell has a kind, a value, and dependencies on other cells.',
    },
    {
      id: 'quilt-federation',
      text: 'Quilt sheets can federate across tiers: ESP32, Jetson, Codespace, Cloudflare Worker, and Server. The same cell graph runs everywhere. Cells are addressable by URI like quilt://instance/sheet#cell.',
    },
    {
      id: 'rag-intro',
      text: 'RAG stands for Retrieval-Augmented Generation. It combines a retriever (which finds relevant documents) with a generator (which produces an answer using those documents).',
    },
  ];

  console.log('Ingesting documents...');
  const count = await pipeline.ingest(docs);
  console.log(`Ingested ${count} chunks.`);

  // Query
  const questions = [
    'What is Quilt?',
    'How many cell kinds does Quilt have?',
    'What does RAG stand for?',
  ];

  for (const q of questions) {
    console.log(`\nQ: ${q}`);
    const result = await pipeline.query(q);
    console.log(`A: ${result.answer}`);
    console.log(`Citations: ${result.citations.map((c) => c.id).join(', ')}`);
    console.log(`Latency: ${result.latencyMs}ms`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
