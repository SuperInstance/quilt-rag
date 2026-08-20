/**
 * Loader cell — loads documents from a source.
 *
 * Built-in loaders: file, URL, S3, R2.
 * Each loader has the same interface, so any source can be swapped for any other.
 */
import { readFile, readdir, stat } from 'node:fs/promises';
import { join, extname } from 'node:path';
import type { Document, Loader } from '../types.js';

/** File loader — loads .txt, .md, .html, .json files. */
export class FileLoader implements Loader {
  async load(source: string): Promise<Document[]> {
    const stats = await stat(source);
    if (stats.isDirectory()) {
      const files = await readdir(source);
      const docs: Document[] = [];
      for (const f of files) {
        const path = join(source, f);
        const s = await stat(path);
        if (s.isFile()) docs.push(...await this.loadFile(path));
      }
      return docs;
    }
    return this.loadFile(source);
  }

  private async loadFile(path: string): Promise<Document[]> {
    const ext = extname(path).toLowerCase();
    const text = await readFile(path, 'utf8');
    switch (ext) {
      case '.json':
        try {
          const j = JSON.parse(text);
          if (Array.isArray(j)) {
            return j.map((d, i) => ({ id: `${path}#${i}`, text: typeof d === 'string' ? d : JSON.stringify(d) }));
          }
          return [{ id: path, text: JSON.stringify(j) }];
        } catch {
          return [{ id: path, text }];
        }
      case '.md':
      case '.txt':
      case '.html':
        return [{ id: path, text }];
      default:
        return [{ id: path, text }];
    }
  }
}

/** URL loader — fetches a URL and parses it. */
export class UrlLoader implements Loader {
  async load(source: string): Promise<Document[]> {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`URL load failed: ${res.status} ${source}`);
    const text = await res.text();
    return [{ id: source, text, metadata: { source, contentType: res.headers.get('content-type') } }];
  }
}

/** S3 loader — loads from an S3 bucket. */
export class S3Loader implements Loader {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(private client: any, private bucket: string) {}
  async load(source: string): Promise<Document[]> {
    const cmd = new (this.client.GetObjectCommand ?? this.client.GetObjectCommand)({
      Bucket: this.bucket,
      Key: source,
    });
    const res = await this.client.send(cmd);
    const text = await res.Body.transformToString();
    return [{ id: `s3://${this.bucket}/${source}`, text }];
  }
}

/** R2 loader — loads from a Cloudflare R2 bucket. */
export class R2Loader implements Loader {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(private binding: any, private bucketName: string) {}
  async load(source: string): Promise<Document[]> {
    const obj = await this.binding.get(source);
    if (!obj) throw new Error(`R2 object not found: ${source}`);
    const text = await obj.text();
    return [{ id: `r2://${this.bucketName}/${source}`, text }];
  }
}
