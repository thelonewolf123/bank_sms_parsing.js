import { kmeans } from "ml-kmeans";
import { distance } from "ml-distance";
import { tokenize } from "../utils/textPreprocessor.js";

const euclidean = distance.euclidean;

export interface ClusterResult {
  clusters: number[];
  centroids: number[][];
}

/**
 * Build vocabulary from all documents
 */
function buildVocabulary(documents: string[][]): Map<string, number> {
  const vocab = new Map<string, number>();
  let index = 0;

  for (const doc of documents) {
    for (const token of doc) {
      if (!vocab.has(token)) {
        vocab.set(token, index++);
      }
    }
  }

  return vocab;
}

/**
 * Convert tokenized document to a bag-of-words vector
 */
function documentToVector(
  tokens: string[],
  vocabulary: Map<string, number>
): number[] {
  const vector = new Array(vocabulary.size).fill(0);

  for (const token of tokens) {
    const idx = vocabulary.get(token);
    if (idx !== undefined) {
      vector[idx]++;
    }
  }

  // Normalize the vector
  const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  if (norm > 0) {
    for (let i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
  }

  return vector;
}

/**
 * Vectorize all texts into numerical feature vectors
 */
export function vectorizeTexts(texts: string[]): {
  vectors: number[][];
  vocabulary: Map<string, number>;
} {
  const documents = texts.map((text) => tokenize(text));
  const vocabulary = buildVocabulary(documents);
  const vectors = documents.map((doc) => documentToVector(doc, vocabulary));

  return { vectors, vocabulary };
}

/**
 * Perform K-means clustering on text data
 */
export function clusterTexts(
  texts: string[],
  numClusters?: number
): ClusterResult {
  const { vectors } = vectorizeTexts(texts);

  // Auto-determine number of clusters if not specified
  // Use sqrt(n/2) as a heuristic, with min 2 and max 100
  const k =
    numClusters ??
    Math.min(100, Math.max(2, Math.floor(Math.sqrt(texts.length / 2))));

  const result = kmeans(vectors, k, {
    initialization: "kmeans++",
    maxIterations: 100
  });

  return {
    clusters: result.clusters,
    centroids: result.centroids
  };
}

/**
 * Find K nearest neighbors for a given vector
 */
export function findKNearestNeighbors(
  queryVector: number[],
  allVectors: number[][],
  k: number
): { index: number; distance: number }[] {
  const distances = allVectors.map((vec, index) => ({
    index,
    distance: euclidean(queryVector, vec)
  }));

  distances.sort((a, b) => a.distance - b.distance);
  return distances.slice(0, k);
}

/**
 * Group similar texts using DBSCAN-like approach based on distance threshold
 */
export function groupTextsByDistance(
  texts: string[],
  distanceThreshold: number = 0.8
): number[] {
  const { vectors } = vectorizeTexts(texts);
  const labels = new Array(texts.length).fill(-1);
  let currentCluster = 0;

  for (let i = 0; i < vectors.length; i++) {
    if (labels[i] !== -1) continue;

    // Start a new cluster
    labels[i] = currentCluster;

    // Find all points within distance threshold
    for (let j = i + 1; j < vectors.length; j++) {
      if (labels[j] !== -1) continue;

      const distance = euclidean(vectors[i], vectors[j]);
      if (distance <= distanceThreshold) {
        labels[j] = currentCluster;
      }
    }

    currentCluster++;
  }

  return labels;
}
