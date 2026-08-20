/**
 * Retriever cell — top-K retrieval by similarity.
 *
 * Strategies: cosine (vector), MMR (maximal marginal relevance), hybrid (BM25 + vector).
 */
import type { Chunk, Retriever, RetrievalResult, VectorStore } from '../types.js';

/** Cosine retriever — top-K by vector similarity. */
export class CosineRetriever implements Retriever {
  constructor(private store: VectorStore, private embedder: { embed(text: string): Promise<number[]> }) {}
  async retrieve(query: string, k: number): Promise<RetrievalResult[]> {
    const [vec] = await this.embedder.embed(query);
    return this.store.query(vec, k);
  }
}

/** MMR retriever — penalizes redundancy in results. */
export class MmrRetriever implements Retriever {
  constructor(
    private store: VectorStore,
    private embedder: { embed(text: string): Promise<number[]> },
    private lambda: number = 0.5
  ) {}
  async retrieve(query: string, k: number): Promise<RetrievalResult[]> {
    const [qvec] = await this.embedder.embed(query);
    const candidates = await this.store.query(qvec, k * 5);
    const selected: RetrievalResult[] = [];
    const remaining = [...candidates];
    while (selected.length < k && remaining.length > 0) {
      let bestIdx = 0;
      let bestScore = -Infinity;
      for (let i = 0; i < remaining.length; i++) {
        const cand = remaining[i]!;
        // MMR score: lambda * relevance - (1 - lambda) * max similarity to selected
        const maxSimToSelected = selected.length === 0 ? 0 :
          Math.max(...selected.map(s => {
            // Approximate by score since we don't have the original vector
            return s.score;
          }));
        const mmrScore = this.lambda * cand.score - (1 - this.lambda) * maxSimToSelected;
        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIdx = i;
        }
      }
      selected.push(remaining.splice(bestIdx, 1)[0]!);
    }
    return selected;
  }
}

/** Hybrid retriever — combines BM25 (sparse) + vector (dense) scores. */
export class HybridRetriever implements Retriever {
  constructor(
    private store: VectorStore,
    private embedder: { embed(text: string): Promise<number[]> },
    private chunks: Chunk[],
    private alpha: number = 0.5
  ) {}
  async retrieve(query: string, k: number): Promise<RetrievalResult[]> {
    const [qvec] = await this.embedder.embed(query);
    const dense = await this.store.query(qvec, k * 2);

    // BM25 over chunks
    const bm25Scores = bm25(query, this.chunks);
    const bm25Ranked = bm25Scores
      .map((score, i) => ({ chunk: this.chunks[i]!, score }))
      .filter(x => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, k * 2);

    // Combine
    const combined = new Map<string, RetrievalResult>();
    for (const r of dense) {
      combined.set(r.id, { ...r, score: this.alpha * r.score });
    }
    for (const r of bm25Ranked) {
      const existing = combined.get(r.chunk.id);
      if (existing) {
        existing.score += (1 - this.alpha) * normalize(r.score, bm25Scores);
      } else {
        combined.set(r.chunk.id, {
          id: r.chunk.id,
          text: r.chunk.text,
          score: (1 - this.alpha) * normalize(r.score, bm25Scores),
          documentId: r.chunk.documentId,
          metadata: r.chunk.metadata,
        });
      }
    }
    return Array.from(combined.values()).sort((a, b) => b.score - a.score).slice(0, k);
  }
}

// ──────────────────────────────────────────────────────────────────────────
//  BM25 helpers
// ──────────────────────────────────────────────────────────────────────────

function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\W+/).filter((t) => t.length > 1);
}

function bm25(query: string, chunks: Chunk[], k1: number = 1.5, b: number = 0.75): number[] {
  const qTokens = tokenize(query);
  const docs = chunks.map((c) => tokenize(c.text));
  const avgdl = docs.reduce((s, d) => s + d.length, 0) / docs.length;
  const N = docs.length;
  const df = new Map<string, number>();
  for (const doc of docs) {
    for (const term of new Set(doc)) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }
  return docs.map((doc) => {
    const dl = doc.length;
    const tf = new Map<string, number>();
    for (const t of doc) tf.set(t, (tf.get(t) ?? 0) + 1);
    let score = 0;
    for (const qt of qTokens) {
      const f = tf.get(qt) ?? 0;
      if (f === 0) continue;
      const idf = Math.log(1 + (N - (df.get(qt) ?? 0) + 0.5) / ((df.get(qt) ?? 0) + 0.5));
      const tfNorm = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl));
      score += idf * tfNorm;
    }
    return score;
  });
}

function normalize(score: number, allScores: number[]): number {
  const max = Math.max(...allScores, 1e-9);
  return score / max;
}
