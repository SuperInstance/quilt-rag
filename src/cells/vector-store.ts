/**
 * Vector store cell — stores and queries vectors.
 *
 * 5 backends: in-memory, Cloudflare Vectorize, Pinecone, Qdrant, pgvector.
 * All conform to the same `VectorStore` interface.
 */
import type { Embedding, RetrievalResult, VectorStore } from '../types.js';

/** In-memory vector store — for tests and ephemeral workloads. */
export class MemoryVectorStore implements VectorStore {
  private store: Embedding[] = [];

  async upsert(embeddings: Embedding[]): Promise<void> {
    for (const e of embeddings) {
      const idx = this.store.findIndex((s) => s.id === e.id);
      if (idx >= 0) this.store[idx] = e;
      else this.store.push(e);
    }
  }

  async query(vector: number[], k: number, filter?: Record<string, unknown>): Promise<RetrievalResult[]> {
    const results: RetrievalResult[] = [];
    for (const e of this.store) {
      if (filter && !matchesFilter(e.metadata, filter)) continue;
      results.push({
        id: e.id,
        text: e.metadata?.text as string ?? '',
        score: cosineSimilarity(vector, e.vector),
        documentId: e.metadata?.documentId as string ?? '',
        metadata: e.metadata,
      });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, k);
  }

  async delete(ids: string[]): Promise<void> {
    this.store = this.store.filter((e) => !ids.includes(e.id));
  }

  async count(): Promise<number> { return this.store.length; }
}

/** Cloudflare Vectorize store. */
export class VectorizeStore implements VectorStore {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(private index: any) {}

  async upsert(embeddings: Embedding[]): Promise<void> {
    await this.index.upsert(embeddings.map((e) => ({
      id: e.id,
      values: e.vector,
      metadata: e.metadata,
    })));
  }

  async query(vector: number[], k: number, filter?: Record<string, unknown>): Promise<RetrievalResult[]> {
    const res = await this.index.query(vector, { topK: k, filter, returnMetadata: 'all' });
    return (res.matches ?? []).map((m: { id: string; score: number; metadata?: Record<string, unknown> }) => ({
      id: m.id,
      text: (m.metadata?.text as string) ?? '',
      score: m.score,
      documentId: (m.metadata?.documentId as string) ?? '',
      metadata: m.metadata,
    }));
  }

  async delete(ids: string[]): Promise<void> {
    await this.index.deleteByIds(ids);
  }

  async count(): Promise<number> {
    const res = await this.index.describe();
    return res.vectorsCount ?? 0;
  }
}

/** Pinecone store. */
export class PineconeStore implements VectorStore {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(private index: any) {}

  async upsert(embeddings: Embedding[]): Promise<void> {
    await this.index.upsert(embeddings.map((e) => ({
      id: e.id,
      values: e.vector,
      metadata: e.metadata,
    })));
  }

  async query(vector: number[], k: number, filter?: Record<string, unknown>): Promise<RetrievalResult[]> {
    const res = await this.index.query({ topK: k, vector, includeMetadata: true, filter });
    return (res.matches ?? []).map((m: { id: string; score: number; metadata?: Record<string, unknown> }) => ({
      id: m.id,
      text: (m.metadata?.text as string) ?? '',
      score: m.score,
      documentId: (m.metadata?.documentId as string) ?? '',
      metadata: m.metadata,
    }));
  }

  async delete(ids: string[]): Promise<void> {
    await this.index.deleteMany(ids);
  }

  async count(): Promise<number> {
    const stats = await this.index.describeIndexStats();
    return stats.totalRecordCount ?? 0;
  }
}

/** Qdrant store. */
export class QdrantStore implements VectorStore {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(private client: any, private collection: string) {}

  async upsert(embeddings: Embedding[]): Promise<void> {
    await this.client.upsert(this.collection, {
      points: embeddings.map((e) => ({ id: e.id, vector: e.vector, payload: e.metadata })),
    });
  }

  async query(vector: number[], k: number, filter?: Record<string, unknown>): Promise<RetrievalResult[]> {
    const res = await this.client.search(this.collection, {
      vector,
      limit: k,
      filter: filter ? { must: Object.entries(filter).map(([k, v]) => ({ key: k, match: { value: v } })) } : undefined,
      with_payload: true,
    });
    return res.map((m: { id: string; score: number; payload?: Record<string, unknown> }) => ({
      id: String(m.id),
      text: (m.payload?.text as string) ?? '',
      score: m.score,
      documentId: (m.payload?.documentId as string) ?? '',
      metadata: m.payload,
    }));
  }

  async delete(ids: string[]): Promise<void> {
    await this.client.delete(this.collection, { points: ids });
  }

  async count(): Promise<number> {
    const res = await this.client.count(this.collection);
    return res.count ?? 0;
  }
}

/** pgvector store. */
export class PgVectorStore implements VectorStore {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(private pool: any, private table: string = 'embeddings') {}

  async upsert(embeddings: Embedding[]): Promise<void> {
    for (const e of embeddings) {
      const vec = `[${e.vector.join(',')}]`;
      await this.pool.query(
        `INSERT INTO ${this.table} (id, vector, metadata) VALUES ($1, $2::vector, $3)
         ON CONFLICT (id) DO UPDATE SET vector = $2::vector, metadata = $3`,
        [e.id, vec, JSON.stringify(e.metadata ?? {})]
      );
    }
  }

  async query(vector: number[], k: number, filter?: Record<string, unknown>): Promise<RetrievalResult[]> {
    const vec = `[${vector.join(',')}]`;
    const filterClause = filter
      ? `AND metadata @> $${3}::jsonb`
      : '';
    const params: unknown[] = [vec, k];
    if (filter) params.push(JSON.stringify(filter));
    const res = await this.pool.query(
      `SELECT id, 1 - (vector <=> $1::vector) AS score, metadata
       FROM ${this.table}
       WHERE 1=1 ${filterClause}
       ORDER BY vector <=> $1::vector
       LIMIT $2`,
      params
    );
    return res.rows.map((row: { id: string; score: number; metadata: Record<string, unknown> }) => ({
      id: row.id,
      text: (row.metadata?.text as string) ?? '',
      score: row.score,
      documentId: (row.metadata?.documentId as string) ?? '',
      metadata: row.metadata,
    }));
  }

  async delete(ids: string[]): Promise<void> {
    await this.pool.query(`DELETE FROM ${this.table} WHERE id = ANY($1)`, [ids]);
  }

  async count(): Promise<number> {
    const res = await this.pool.query(`SELECT COUNT(*)::int AS count FROM ${this.table}`);
    return res.rows[0]?.count ?? 0;
  }
}

// ──────────────────────────────────────────────────────────────────────────
//  Helpers
// ──────────────────────────────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

function matchesFilter(meta: Record<string, unknown> | undefined, filter: Record<string, unknown>): boolean {
  if (!meta) return false;
  for (const [k, v] of Object.entries(filter)) {
    if (meta[k] !== v) return false;
  }
  return true;
}
