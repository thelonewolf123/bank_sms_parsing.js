import {
  pipeline,
  ZeroShotClassificationPipeline
} from "@huggingface/transformers";

const CATEGORIES = [
  "bank_transaction_sms",
  "failed_bank_transaction_sms",

  "upi_transaction_sms",
  "failed_upi_transaction_sms",

  "promotional",
  "other"
] as const;

export type Category = (typeof CATEGORIES)[number];

export interface ClassificationResult {
  text: string;
  category: Category;
  confidence: number;
  allScores: Record<Category, number>;
}

let classifierInstance: ZeroShotClassificationPipeline | null = null;

async function getClassifier(): Promise<ZeroShotClassificationPipeline> {
  if (!classifierInstance) {
    classifierInstance = await pipeline(
      "zero-shot-classification",
      "Xenova/mobilebert-uncased-mnli"
    );
  }
  return classifierInstance;
}

export async function classifyText(
  text: string
): Promise<ClassificationResult> {
  const classifier = await getClassifier();

  const output = await classifier(text, [...CATEGORIES], {
    multi_label: false
  });

  // Handle both single result and array result
  const result = Array.isArray(output) ? output[0] : output;

  const scores: Record<string, number> = {};
  const labels = result.labels as string[];
  const resultScores = result.scores as number[];

  labels.forEach((label, i) => {
    scores[label] = resultScores[i];
  });

  return {
    text,
    category: labels[0] as Category,
    confidence: resultScores[0],
    allScores: scores as Record<Category, number>
  };
}

export async function classifyBatch(
  texts: string[]
): Promise<ClassificationResult[]> {
  const results: ClassificationResult[] = [];

  for (const text of texts) {
    results.push(await classifyText(text));
  }

  return results;
}

export { CATEGORIES };
