import natural from "natural";

const tokenizer = new natural.WordTokenizer();
const stopwords = new Set(natural.stopwords);

export function tokenize(text: string): string[] {
  const tokens = tokenizer.tokenize(text.toLowerCase()) || [];
  return tokens.filter((token) => token.length > 1 && !stopwords.has(token));
}

export function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/\d+/g, "NUM")
    .replace(/xx+\d*/gi, "MASKED")
    .replace(/[^a-z\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
