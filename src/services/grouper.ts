import { TfIdf } from "../lib/tfidf";
import { SmsRecord } from "../utils/csvParser";

export interface SmsGroup {
  centroidIndex: number;
  members: number[];
  sample: SmsRecord;
}

export function groupSmsBySimilarity(
  records: SmsRecord[],
  similarityThreshold = 0.5
): SmsGroup[] {
  const tfidf = new TfIdf();
  const texts = records.map((r) => r.text);
  tfidf.addDocuments(texts);

  // Compute all TF-IDF vectors
  const vectors: Map<string, number>[] = [];
  for (let i = 0; i < tfidf.documentCount; i++) {
    vectors.push(tfidf.getTfIdfVector(i));
  }

  // Simple clustering: assign each document to the first group it's similar enough to
  const groups: SmsGroup[] = [];
  const assigned = new Set<number>();

  for (let i = 0; i < vectors.length; i++) {
    if (assigned.has(i)) continue;

    // Start a new group with this document as centroid
    const group: SmsGroup = {
      centroidIndex: i,
      members: [i],
      sample: records[i]
    };
    assigned.add(i);

    // Find all similar documents
    for (let j = i + 1; j < vectors.length; j++) {
      if (assigned.has(j)) continue;

      const similarity = tfidf.cosineSimilarity(vectors[i], vectors[j]);
      if (similarity >= similarityThreshold) {
        group.members.push(j);
        assigned.add(j);
      }
    }

    groups.push(group);
  }

  return groups;
}

export function sampleFromGroups(
  groups: SmsGroup[],
  records: SmsRecord[],
  samplesPerGroup = 1
): { group: SmsGroup; samples: SmsRecord[] }[] {
  return groups.map((group) => {
    const sampleIndices = group.members.slice(0, samplesPerGroup);
    return {
      group,
      samples: sampleIndices.map((idx) => records[idx])
    };
  });
}
