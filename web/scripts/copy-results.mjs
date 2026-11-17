// Copy the generated experiment results into the console's static files.
// The numbers are never typed by hand: they come from docs/results/, which training writes.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(root, "../docs/results/experiments.json");
const target = resolve(root, "public/data/experiments.json");
mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
console.log(`copied ${source} -> ${target}`);
