import { parseCsv } from "./utils/csvParser";
import { groupSmsBySimilarity, sampleFromGroups } from "./services/grouper";
import {
  classifyText,
  ClassificationResult,
  Category
} from "./services/classifier";
import path from "path";

interface GroupClassification {
  groupId: number;
  memberCount: number;
  sampleText: string;
  classification: ClassificationResult;
}

async function main() {
  const csvPath = path.join(__dirname, "../data/sms_filtered.csv");

  console.log("Loading SMS data...");
  const records = parseCsv(csvPath);
  console.log(`Loaded ${records.length} SMS records`);

  console.log("\nGrouping SMS by TF-IDF similarity...");
  const groups = groupSmsBySimilarity(records, 0.4);
  console.log(`Created ${groups.length} groups`);

  console.log("\nSampling from groups and classifying...");
  const sampledGroups = sampleFromGroups(groups, records);

  const results: GroupClassification[] = [];
  const categoryStats: Record<Category, { groups: number; messages: number }> =
    {
      bank_transaction_sms: { groups: 0, messages: 0 },
      failed_bank_transaction_sms: { groups: 0, messages: 0 },
      promotional: { groups: 0, messages: 0 },
      other: { groups: 0, messages: 0 }
    };

  for (let i = 0; i < sampledGroups.length; i++) {
    const { group, samples } = sampledGroups[i];
    const sampleText = samples[0].text;

    process.stdout.write(
      `\rClassifying group ${i + 1}/${sampledGroups.length}...`
    );

    const classification = await classifyText(sampleText);

    results.push({
      groupId: i,
      memberCount: group.members.length,
      sampleText,
      classification
    });

    categoryStats[classification.category].groups++;
    categoryStats[classification.category].messages += group.members.length;
  }

  console.log("\n\n=== Classification Results ===\n");

  console.log("Category Statistics:");
  console.log("-".repeat(60));
  for (const [category, stats] of Object.entries(categoryStats)) {
    console.log(
      `${category.padEnd(30)} Groups: ${stats.groups
        .toString()
        .padStart(5)} | Messages: ${stats.messages.toString().padStart(6)}`
    );
  }

  console.log("\n\nSample Classifications:");
  console.log("-".repeat(80));

  // Show a few examples from each category
  const shownPerCategory: Record<string, number> = {};
  for (const result of results) {
    const cat = result.classification.category;
    shownPerCategory[cat] = (shownPerCategory[cat] || 0) + 1;

    if (shownPerCategory[cat] <= 2) {
      console.log(
        `\n[${cat}] (confidence: ${(
          result.classification.confidence * 100
        ).toFixed(1)}%)`
      );
      console.log(`Group size: ${result.memberCount}`);
      console.log(`Sample: ${result.sampleText.substring(0, 100)}...`);
    }
  }

  return results;
}

main().catch(console.error);
