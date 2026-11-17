// Steps of the contribution waterfall, from the model's baseline to this transaction's
// margin. Everything is in log-odds, the model's own units: contributions add up exactly
// there, and would not in probabilities.

export type Contribution = { feature: string; contribution: number };

export type WaterfallStep = {
  label: string;
  /** Where the bar starts and ends, in log-odds. */
  start: number;
  end: number;
  /** The step's size; the baseline and total bars carry their own value. */
  value: number;
  kind: "baseline" | "up" | "down" | "total";
};

/**
 * Build the waterfall: the baseline, the ``top`` largest contributions by size, one bar
 * for all the rest, and the final margin.
 */
export function buildWaterfall(
  bias: number,
  contributions: Contribution[],
  top = 8,
): WaterfallStep[] {
  const ranked = [...contributions].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution),
  );
  const shown = ranked.slice(0, top);
  const rest = ranked.slice(top).reduce((sum, c) => sum + c.contribution, 0);
  const steps: WaterfallStep[] = [
    { label: "Baseline", start: 0, end: bias, value: bias, kind: "baseline" },
  ];
  let level = bias;
  const add = (label: string, value: number) => {
    steps.push({
      label,
      start: level,
      end: level + value,
      value,
      kind: value >= 0 ? "up" : "down",
    });
    level += value;
  };
  for (const c of shown) add(c.feature, c.contribution);
  if (ranked.length > top) add(`${ranked.length - top} other features`, rest);
  steps.push({ label: "This transaction", start: 0, end: level, value: level, kind: "total" });
  return steps;
}
