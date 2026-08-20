/**
 * Chunker cell — splits documents into chunks.
 *
 * Built-in strategies: sentence, paragraph, token-window, semantic.
 * Each strategy has tradeoffs; pick based on your domain.
 */
import type { Chunk, Chunker, Document } from '../types.js';

/** Sentence chunker — splits on sentence boundaries. */
export class SentenceChunker implements Chunker {
  async chunk(doc: Document, options: { maxTokens?: number; overlap?: number } = {}): Promise<Chunk[]> {
    const maxTokens = options.maxTokens ?? 512;
    const sentences = doc.text.split(/(?<=[.!?])\s+/);
    const chunks: Chunk[] = [];
    let buf: string[] = [];
    let tokenCount = 0;
    for (const s of sentences) {
      const tokens = estimateTokens(s);
      if (tokenCount + tokens > maxTokens && buf.length) {
        chunks.push({ id: `${doc.id}#${chunks.length}`, text: buf.join(' '), documentId: doc.id, index: chunks.length });
        buf = [];
        tokenCount = 0;
      }
      buf.push(s);
      tokenCount += tokens;
    }
    if (buf.length) {
      chunks.push({ id: `${doc.id}#${chunks.length}`, text: buf.join(' '), documentId: doc.id, index: chunks.length });
    }
    return chunks;
  }
}

/** Paragraph chunker — splits on double newlines. */
export class ParagraphChunker implements Chunker {
  async chunk(doc: Document, options: { maxTokens?: number; overlap?: number } = {}): Promise<Chunk[]> {
    const maxTokens = options.maxTokens ?? 1024;
    const paragraphs = doc.text.split(/\n\s*\n/);
    const chunks: Chunk[] = [];
    let buf: string[] = [];
    let tokenCount = 0;
    for (const p of paragraphs) {
      const tokens = estimateTokens(p);
      if (tokenCount + tokens > maxTokens && buf.length) {
        chunks.push({ id: `${doc.id}#${chunks.length}`, text: buf.join('\n\n'), documentId: doc.id, index: chunks.length });
        buf = [];
        tokenCount = 0;
      }
      buf.push(p);
      tokenCount += tokens;
    }
    if (buf.length) {
      chunks.push({ id: `${doc.id}#${chunks.length}`, text: buf.join('\n\n'), documentId: doc.id, index: chunks.length });
    }
    return chunks;
  }
}

/** Token window chunker — fixed token count with overlap. */
export class TokenWindowChunker implements Chunker {
  async chunk(doc: Document, options: { maxTokens?: number; overlap?: number } = {}): Promise<Chunk[]> {
    const maxTokens = options.maxTokens ?? 512;
    const overlap = options.overlap ?? 50;
    const tokens = tokenize(doc.text);
    const chunks: Chunk[] = [];
    let i = 0;
    while (i < tokens.length) {
      const end = Math.min(i + maxTokens, tokens.length);
      const text = tokens.slice(i, end).map((t) => t.text).join(' ');
      chunks.push({ id: `${doc.id}#${chunks.length}`, text, documentId: doc.id, index: chunks.length });
      if (end === tokens.length) break;
      i = end - overlap;
    }
    return chunks;
  }
}

/** Semantic chunker — chunks based on embedding similarity breaks. */
export class SemanticChunker implements Chunker {
  constructor(private embedFn: (text: string) => Promise<number[]>, private threshold: number = 0.5) {}
  async chunk(doc: Document, options: { maxTokens?: number; overlap?: number } = {}): Promise<Chunk[]> {
    const maxTokens = options.maxTokens ?? 512;
    const sentences = doc.text.split(/(?<=[.!?])\s+/);
    if (sentences.length <= 1) {
      return [{ id: `${doc.id}#0`, text: doc.text, documentId: doc.id, index: 0 }];
    }
    const chunks: Chunk[] = [];
    let buf: string[] = [];
    let bufTokens = 0;
    let prevVec: number[] | null = null;
    for (const s of sentences) {
      const vec = await this.embedFn(s);
      const sim = prevVec ? cosineSimilarity(prevVec, vec) : 1;
      prevVec = vec;
      const tokens = estimateTokens(s);
      const shouldBreak = sim < this.threshold && buf.length > 0;
      if (shouldBreak || bufTokens + tokens > maxTokens) {
        if (buf.length) {
          chunks.push({ id: `${doc.id}#${chunks.length}`, text: buf.join(' '), documentId: doc.id, index: chunks.length });
          buf = [];
          bufTokens = 0;
        }
      }
      buf.push(s);
      bufTokens += tokens;
    }
    if (buf.length) {
      chunks.push({ id: `${doc.id}#${chunks.length}`, text: buf.join(' '), documentId: doc.id, index: chunks.length });
    }
    return chunks;
  }
}

// ──────────────────────────────────────────────────────────────────────────
//  Helpers
// ──────────────────────────────────────────────────────────────────────────

/** Rough token estimate: ~4 chars per token for English. */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Naive word tokenizer. */
function tokenize(text: string): { text: string }[] {
  return text.split(/\s+/).filter(Boolean).map((t) => ({ text: t }));
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}
