// What a pair of thresholds would have done on the validation slice. Prices come from
// the service, which uses the same function that chose the released thresholds; this
// module only reads them.
import type { ThresholdLab } from "../api/client";

export type OperatingPoint = {
  tReview: number;
  tBlock: number;
  /** Among transactions sent to review or block. */
  precision: number | null;
  recall: number | null;
  dollarRecall: number | null;
  alertsPerDay: number;
  blocksPerDay: number;
  reviewShare: number;
  costPer1000: number;
  withinCapacity: boolean;
  isReleased: boolean;
};

/**
 * Describe the pair ``(review, block)``, both indices into ``lab.points``.
 * ``block`` below ``review`` is raised to ``review``.
 */
export function operatingPoint(lab: ThresholdLab, review: number, block: number): OperatingPoint {
  const b = Math.max(block, review);
  const reviewPoint = lab.points[review];
  const blockPoint = lab.points[b];
  const pair = lab.pairs.find((p) => p.review === review && p.block === b);
  if (!reviewPoint || !blockPoint || !pair) {
    throw new RangeError(`no threshold pair (${review}, ${b})`);
  }
  return {
    tReview: reviewPoint.threshold,
    tBlock: blockPoint.threshold,
    precision: reviewPoint.precision,
    recall: reviewPoint.recall,
    dollarRecall: reviewPoint.dollar_recall,
    alertsPerDay: reviewPoint.alerts_per_day,
    blocksPerDay: blockPoint.count / lab.days,
    reviewShare: pair.review_share,
    costPer1000: pair.cost_per_1000,
    withinCapacity: pair.within_capacity,
    isReleased: review === lab.current_review && b === lab.current_block,
  };
}

/** Display a threshold; 1.01 is the "never" sentinel above any probability. */
export function thresholdLabel(threshold: number): string {
  return threshold > 1 ? "never" : threshold.toFixed(4);
}
