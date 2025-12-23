import { tokenize } from "../utils/textPreprocessor";

export class TfIdf {
  private documents: string[][] = [];
  private idfCache: Map<string, number> = new Map();

  addDocument(text: string): void {
    this.documents.push(tokenize(text));
    this.idfCache.clear();
  }

  addDocuments(texts: string[]): void {
    texts.forEach((text) => this.addDocument(text));
  }

  private computeTf(tokens: string[]): Map<string, number> {
    const tf = new Map<string, number>();
    tokens.forEach((token) => {
      tf.set(token, (tf.get(token) || 0) + 1);
    });
    // Normalize by document length
    tokens.length > 0 &&
      tf.forEach((count, token) => tf.set(token, count / tokens.length));
    return tf;
  }

  private computeIdf(term: string): number {
    if (this.idfCache.has(term)) {
      return this.idfCache.get(term)!;
    }
    const docsWithTerm = this.documents.filter((doc) =>
      doc.includes(term)
    ).length;
    const idf =
      docsWithTerm > 0 ? Math.log(this.documents.length / docsWithTerm) + 1 : 0;
    this.idfCache.set(term, idf);
    return idf;
  }

  getTfIdfVector(docIndex: number): Map<string, number> {
    const tokens = this.documents[docIndex];
    const tf = this.computeTf(tokens);
    const tfidf = new Map<string, number>();

    tf.forEach((tfValue, term) => {
      tfidf.set(term, tfValue * this.computeIdf(term));
    });

    return tfidf;
  }

  cosineSimilarity(
    vec1: Map<string, number>,
    vec2: Map<string, number>
  ): number {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    vec1.forEach((val, key) => {
      norm1 += val * val;
      if (vec2.has(key)) {
        dotProduct += val * vec2.get(key)!;
      }
    });

    vec2.forEach((val) => {
      norm2 += val * val;
    });

    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }

  get documentCount(): number {
    return this.documents.length;
  }
}
