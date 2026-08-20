/**
 * Generator + context builder cells.
 *
 * Generator: calls the LLM with a prompt. Returns text + token usage.
 * Context builder: assembles the prompt context from retrieved results.
 */
import type { ContextBuilder, Generator, RetrievalResult } from '../types.js';

/** Default context builder — top-K with token budget. */
export class DefaultContextBuilder implements ContextBuilder {
  async build(_query: string, results: RetrievalResult[], options: { maxTokens?: number } = {}): Promise<string> {
    const maxTokens = options.maxTokens ?? 2048;
    const parts: string[] = [];
    let tokens = 0;
    for (let i = 0; i < results.length; i++) {
      const r = results[i]!;
      const t = estimateTokens(r.text);
      if (tokens + t > maxTokens && parts.length > 0) break;
      parts.push(`[${i + 1}] ${r.text}`);
      tokens += t;
    }
    return parts.join('\n\n');
  }
}

/** OpenAI-compatible generator. */
export class OpenAIGenerator implements Generator {
  constructor(private apiKey: string, private modelName: string = 'gpt-4o-mini') {}
  async generate(prompt: string, options: { system?: string; temperature?: number; maxTokens?: number } = {}): Promise<{ text: string; tokensUsed?: number }> {
    const messages: Array<{ role: string; content: string }> = [];
    if (options.system) messages.push({ role: 'system', content: options.system });
    messages.push({ role: 'user', content: prompt });
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.modelName, messages, temperature: options.temperature ?? 0.3, max_tokens: options.maxTokens ?? 1024 }),
    });
    if (!res.ok) throw new Error(`OpenAI generate failed: ${res.status}`);
    const j = await res.json() as { choices: Array<{ message: { content: string } }>; usage?: { total_tokens: number } };
    return { text: j.choices[0]?.message.content ?? '', tokensUsed: j.usage?.total_tokens };
  }
}

/** z.ai generator. */
export class ZaiGenerator implements Generator {
  constructor(private apiKey: string, private modelName: string = 'glm-4.5') {}
  async generate(prompt: string, options: { system?: string; temperature?: number; maxTokens?: number } = {}): Promise<{ text: string; tokensUsed?: number }> {
    const messages: Array<{ role: string; content: string }> = [];
    if (options.system) messages.push({ role: 'system', content: options.system });
    messages.push({ role: 'user', content: prompt });
    const res = await fetch('https://api.z.ai/api/paas/v4/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.modelName, messages, temperature: options.temperature ?? 0.3, max_tokens: options.maxTokens ?? 1024 }),
    });
    if (!res.ok) throw new Error(`z.ai generate failed: ${res.status}`);
    const j = await res.json() as { choices: Array<{ message: { content: string } }>; usage?: { total_tokens: number } };
    return { text: j.choices[0]?.message.content ?? '', tokensUsed: j.usage?.total_tokens };
  }
}

/** Cloudflare Workers AI generator. */
export class WorkersAIGenerator implements Generator {
  constructor(private accountId: string, private apiToken: string, private modelName: string = '@cf/meta/llama-3.1-8b-instruct') {}
  async generate(prompt: string, options: { system?: string; temperature?: number; maxTokens?: number } = {}): Promise<{ text: string; tokensUsed?: number }> {
    const messages: Array<{ role: string; content: string }> = [];
    if (options.system) messages.push({ role: 'system', content: options.system });
    messages.push({ role: 'user', content: prompt });
    const res = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${this.accountId}/ai/run/${this.modelName}`,
      {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${this.apiToken}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages, temperature: options.temperature ?? 0.3, max_tokens: options.maxTokens ?? 1024 }),
      }
    );
    if (!res.ok) throw new Error(`Workers AI generate failed: ${res.status}`);
    const j = await res.json() as { result: { response: string } };
    return { text: j.result.response };
  }
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}
