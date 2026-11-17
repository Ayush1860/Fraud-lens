// Display formatting. Amounts are dollars; times are ISO 8601 strings in UTC.

const dollars = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const wholeDollars = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const counts = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function money(amount: number): string {
  return dollars.format(amount);
}

export function moneyRounded(amount: number): string {
  return wholeDollars.format(amount);
}

export function count(value: number): string {
  return counts.format(value);
}

/** A share (0 to 1) as a percentage; "n/a" when unknown. */
export function percent(share: number | null | undefined, digits = 1): string {
  return share === null || share === undefined ? "n/a" : `${(share * 100).toFixed(digits)}%`;
}

/** A calibrated probability with three decimals. */
export function score(probability: number): string {
  return probability.toFixed(3);
}

/** A signed contribution in log-odds. */
export function logOdds(value: number): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;
}

/** "45 s", "12 min", "3 h" or "2 d" between ``iso`` and ``now``. */
export function age(iso: string, now: Date = new Date()): string {
  const seconds = Math.max(0, (now.getTime() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return `${Math.floor(seconds)} s`;
  if (seconds < 3_600) return `${Math.floor(seconds / 60)} min`;
  if (seconds < 86_400) return `${Math.floor(seconds / 3_600)} h`;
  return `${Math.floor(seconds / 86_400)} d`;
}

/** An ISO time as "2020-08-14 03:12 UTC". */
export function utc(iso: string): string {
  return `${new Date(iso).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

/** Feature names as words: "amt_ratio_30d" becomes "amt ratio 30d". */
export function featureLabel(name: string): string {
  return name.replaceAll("_", " ");
}
