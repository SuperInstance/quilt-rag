/**
 * Evaluator cells — score RAG outputs.
 *
 * 4 evaluators: relevance, faithfulness, hallucination, context-precision.
 */
import type { RetrievalResult, RAGResult } from '../types.js';

/** Relevance evaluator — did the retrieved chunks match the query? */
export class RelevanceEvaluator {
  score(query: string, results: RetrievalResult[]): { score: number; perChunk: number[] } {
    const qTokens = new Set(query.toLowerCase().split(/\W+/).filter(Boolean));
    const perChunk = results.map((r) => {
      const rTokens = r.text.toLowerCase().split(/\W+/).filter(Boolean);
      const overlap = rTokens.filter((t) => qTokens.has(t)).length;
      return qTokens.size ? overlap / qTokens.size : 0;
    });
    const score = perChunk.length ? perChunk.reduce((s, v) => s + v, 0) / perChunk.length : 0;
    return { score, perChunk };
  }
}

/** Faithfulness evaluator — did the answer only use the retrieved context? */
export class FaithfulnessEvaluator {
  score(answer: string, results: RetrievalResult[]): { score: number; unsupported: string[] } {
    const contextText = results.map((r) => r.text).join(' ').toLowerCase();
    const sentences = answer.split(/[.!?]+/).map((s) => s.trim().toLowerCase()).filter(Boolean);
    const unsupported: string[] = [];
    let supported = 0;
    for (const s of sentences) {
      const words = s.split(/\W+/).filter((w) => w.length > 3);
      const foundInContext = words.some((w) => contextText.includes(w));
      if (foundInContext) supported++;
      else unsupported.push(s);
    }
    return { score: sentences.length ? supported / sentences.length : 1, unsupported };
  }
}

/** Hallucination evaluator — inverse of faithfulness. */
export class HallucinationEvaluator {
  score(answer: string, results: RetrievalResult[]): { score: number; hallucinatedSentences: string[] } {
    const f = new FaithfulnessEvaluator().score(answer, results);
    return { score: 1 - f.score, hallucinatedSentences: f.unsupported };
  }
}

/** Context precision — were the right chunks ranked highest? */
export class ContextPrecisionEvaluator {
  score(relevantIds: string[], results: RetrievalResult[]): { score: number } {
    if (!results.length) return { score: 0 };
    let hits = 0;
    let total = 0;
    for (let i = 0; i < results.length; i++) {
      const r = results[i]!;
      const isRelevant = relevantIds.includes(r.id);
      if (isRelevant) {
        hits++;
        total += hits / (i + 1);
      }
    }
    return { score: relevantIds.length ? total / relevantIds.length : 0 };
  }
}

/** Evaluator cell — runs all four metrics on a RAG result. */
export class RagEvaluatorCell {
  relevance = new RelevanceEvaluator();
  faithfulness = new FaithfulnessEvaluator();
  hallucination = new HallucinationEvaluator();
  contextPrecision = new ContextPrecisionEvaluator();

  evaluate(result: RAGResult, groundTruth: { relevantIds?: string[] } = {}): {
    relevance: number;
    faithfulness: number;
    hallucination: number;
    contextPrecision: number;
  } {
    return {
      relevance: this.relevance.score('', result.citations).score,
      faithfulness: this.faithfulness.score(result.answer, result.citations).score,
      hallucination: this.hallucination.score(result.answer, result.citations).score,
      contextPrecision: this.contextPrecision.score(groundTruth.relevantIds ?? [], result.citations).score,
    };
  }
}
