/**
 * Reranker cell — re-scores the top-K after initial retrieval.
 *
 * 3 implementations: Workers AI BGE, Cohere, local cross-encoder.
 */
import type { Reranker, RetrievalResult } from '../types.js';

/** Workers AI BGE reranker. */
export class BgeReranker implements Reranker {
  constructor(private accountId: string, private apiToken: string) {}
  async rerank(query: string, results: RetrievalResult[], k?: number): Promise<RetrievalResult[]> {
    const topK = k ?? results.length;
    const res = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${this.accountId}/ai/run/@cf/baai/bge-reranker-base`,
      {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${this.apiToken}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, contexts: results.map((r) => r.text) }),
      }
    );
    if (!res.ok) throw new Error(`BGE rerank failed: ${res.status}`);
    const j = await res.json() as { result: Array<{ id: number; score: number }> };
    const scored = j.result.map((r) => ({ ...results[r.id]!, score: r.score }));
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}

/** Cohere reranker. */
export class CohereReranker implements Reranker {
  constructor(private apiKey: string) {}
  async rerank(query: string, results: RetrievalResult[], k?: number): Promise<RetrievalResult[]> {
    const topK = k ?? results.length;
    const res = await fetch('https://api.cohere.ai/v1/rerank', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'rerank-english-v3.0', query, documents: results.map((r) => r.text), top_n: topK }),
    });
    if (!res.ok) throw new Error(`Cohere rerank failed: ${res.status}`);
    const j = await res.json() as { results: Array<{ index: number; relevance_score: number }> };
    return j.results.map((r) => ({ ...results[r.index]!, score: r.relevance_score }));
  }
}

/** Local cross-encoder reranker (Transformers.js). */
export class LocalCrossEncoderReranker implements Reranker {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private model: any = null;
  constructor(private modelName: string = 'Xenova/ms-marco-MiniLM-L-6-v2') {}

  async rerank(query: string, results: RetrievalResult[], k?: number): Promise<RetrievalResult[]> {
    if (!this.model) {
      const { pipeline } = await import('@huggingface/transformers');
      this.model = await pipeline('text-classification', this.modelName);
    }
    const topK = k ?? results.length;
    const scored: RetrievalResult[] = [];
    for (const r of results) {
      const out = await this.model(`query: ${query}`, `passage: ${r.text}`, { topk: 1 });
      const score = out[0]?.score ?? 0;
      scored.push({ ...r, score });
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}
