/**
 * Embedder cell — vectorizes text.
 *
 * 5 providers: Workers AI, OpenAI, Cohere, Voyage, local ONNX.
 * All conform to the same `Embedder` interface.
 */
import type { Embedder } from '../types.js';

/** Workers AI embedder. */
export class WorkersAIEmbedder implements Embedder {
  constructor(private accountId: string, private apiToken: string) {}
  readonly model = '@cf/baai/bge-small-en-v1.5';
  readonly dimensions = 384;

  async embed(text: string | string[]): Promise<number[][]> {
    const inputs = Array.isArray(text) ? text : [text];
    const res = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${this.accountId}/ai/run/${this.model}`,
      {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${this.apiToken}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputs }),
      }
    );
    if (!res.ok) throw new Error(`Workers AI failed: ${res.status}`);
    const j = await res.json() as { result: { data: number[][] } };
    return j.result.data;
  }
}

/** OpenAI embedder. */
export class OpenAIEmbedder implements Embedder {
  constructor(private apiKey: string, private modelName: string = 'text-embedding-3-small') {
    if (modelName === 'text-embedding-3-small') this.dimensions = 1536;
    else if (modelName === 'text-embedding-3-large') this.dimensions = 3072;
    else this.dimensions = 1536;
  }
  readonly model = this.modelName;
  dimensions: number;

  async embed(text: string | string[]): Promise<number[][]> {
    const inputs = Array.isArray(text) ? text : [text];
    const res = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: inputs }),
    });
    if (!res.ok) throw new Error(`OpenAI embed failed: ${res.status}`);
    const j = await res.json() as { data: Array<{ embedding: number[] }> };
    return j.data.map((d) => d.embedding);
  }
}

/** Cohere embedder. */
export class CohereEmbedder implements Embedder {
  constructor(private apiKey: string) {}
  readonly model = 'embed-english-v3.0';
  readonly dimensions = 1024;

  async embed(text: string | string[]): Promise<number[][]> {
    const inputs = Array.isArray(text) ? text : [text];
    const res = await fetch('https://api.cohere.ai/v1/embed', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, texts: inputs, input_type: 'search_document' }),
    });
    if (!res.ok) throw new Error(`Cohere embed failed: ${res.status}`);
    const j = await res.json() as { embeddings: number[][] };
    return j.embeddings;
  }
}

/** Voyage embedder. */
export class VoyageEmbedder implements Embedder {
  constructor(private apiKey: string) {}
  readonly model = 'voyage-3';
  readonly dimensions = 1024;

  async embed(text: string | string[]): Promise<number[][]> {
    const inputs = Array.isArray(text) ? text : [text];
    const res = await fetch('https://api.voyageai.com/v1/embeddings', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: inputs }),
    });
    if (!res.ok) throw new Error(`Voyage embed failed: ${res.status}`);
    const j = await res.json() as { data: Array<{ embedding: number[] }> };
    return j.data.map((d) => d.embedding);
  }
}

/** Local ONNX embedder (Transformers.js). Lazy-loaded so this works in pure JS envs. */
export class LocalOnnxEmbedder implements Embedder {
  readonly model: string;
  readonly dimensions: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private pipeline: any = null;

  constructor(model: string = 'Xenova/all-MiniLM-L6-v2', dimensions: number = 384) {
    this.model = model;
    this.dimensions = dimensions;
  }

  async embed(text: string | string[]): Promise<number[][]> {
    if (!this.pipeline) {
      // Dynamic import to keep the dep optional
      const { pipeline } = await import('@huggingface/transformers');
      this.pipeline = await pipeline('feature-extraction', this.model);
    }
    const inputs = Array.isArray(text) ? text : [text];
    const results: number[][] = [];
    for (const t of inputs) {
      const out = await this.pipeline(t, { pooling: 'mean', normalize: true });
      results.push(Array.from(out.data as Float32Array));
    }
    return results;
  }
}
