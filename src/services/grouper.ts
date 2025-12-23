import { clusterTexts, groupTextsByDistance } from "../lib/clustering.js";
import { SmsRecord } from "../utils/csvParser.js";

export interface SmsGroup {
  centroidIndex: number;
  members: number[];
  sample: SmsRecord;
}

export interface ClusteringOptions {
  method: "kmeans" | "distance";
  numClusters?: number; // For kmeans
  distanceThreshold?: number; // For distance-based grouping
}

/**
 * Group SMS messages by similarity using K-means clustering
 */
export function groupSmsBySimilarity(
  records: SmsRecord[],
  options: ClusteringOptions = { method: "kmeans" }
): SmsGroup[] {
  const texts = records.map((r) => r.text);

  let clusterLabels: number[];

  if (options.method === "kmeans") {
    const result = clusterTexts(texts, options.numClusters);
    clusterLabels = result.clusters;
  } else {
    clusterLabels = groupTextsByDistance(
      texts,
      options.distanceThreshold ?? 0.8
    );
  }

  // Group records by cluster label
  const clusterMap = new Map<number, number[]>();
  clusterLabels.forEach((label, idx) => {
    if (!clusterMap.has(label)) {
      clusterMap.set(label, []);
    }
    clusterMap.get(label)!.push(idx);
  });

  // Convert to SmsGroup format
  const groups: SmsGroup[] = [];
  for (const [, members] of clusterMap) {
    const centroidIndex = members[0]; // Use first member as representative
    groups.push({
      centroidIndex,
      members,
      sample: records[centroidIndex]
    });
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
