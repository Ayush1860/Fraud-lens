// The shape of docs/results/experiments.json, written by the imbalance study, and safe
// readers for the bundle's free-form metrics document.

export type Interval = { mean: number; ci95: [number, number]; seed_std: number };

export type Arm = {
  arm: string;
  description: string;
  summary: Record<string, Interval>;
  pr_curve: { precision: number[]; recall: number[] };
};

export type ExperimentsFile = {
  dataset: string;
  evaluated_on: string;
  eval_rows: number;
  eval_frauds: number;
  seeds: number[];
  n_boot: number;
  arms: Arm[];
};

/** Read a number at ``path`` in an untyped JSON document, or ``null``. */
export function numberAt(document: unknown, ...path: string[]): number | null {
  let value: unknown = document;
  for (const key of path) {
    if (typeof value !== "object" || value === null || !(key in value)) return null;
    value = (value as Record<string, unknown>)[key];
  }
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Points of an arm's precision-recall curve, for plotting. */
export function curvePoints(arm: Arm): { recall: number; precision: number }[] {
  return arm.pr_curve.recall.map((recall, i) => ({
    recall,
    precision: arm.pr_curve.precision[i] ?? Number.NaN,
  }));
}
