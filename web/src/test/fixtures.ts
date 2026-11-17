// Response bodies shaped by the generated API types, so a contract change breaks them.
import type {
  Alert,
  CardTransaction,
  DecisionRecord,
  Explanation,
  ModelInfo,
  MonitoringSummary,
  ThresholdLab,
} from "../api/client";
import type { ExperimentsFile } from "../lib/experiments";

export const alert: Alert = {
  id: 7,
  transaction_id: "tx-7",
  priority: 512.4,
  status: "open",
  decision: "block",
  risk_score: 0.97,
  amount: 528.25,
  category: "shopping_net",
  event_time: "2020-08-14T03:12:44Z",
  created_at: "2020-08-14T03:12:45Z",
  top_reason: "Amount is 8.2 times this card's 30-day average of $64",
};

export const decision: DecisionRecord = {
  transaction_id: "tx-7",
  card_id: "c_0123456789abcdef",
  event_time: "2020-08-14T03:12:44Z",
  amount: 528.25,
  category: "shopping_net",
  features: { amount: 528.25, n_prior_30d: 12 },
  margin: 3.1,
  risk_score: 0.97,
  decision: "block",
  t_review: 0.02,
  t_block: 0.9333,
  model_version: "2026-11-17.1",
  reason_codes: [],
  degraded: false,
  latency_ms: 9,
  created_at: "2020-08-14T03:12:45Z",
};

export const explanation: Explanation = {
  transaction_id: "tx-7",
  model_version: "2026-11-17.1",
  bias: -6.0,
  margin: 3.1,
  risk_score: 0.97,
  contributions: [
    { feature: "amt_ratio_30d", value: 8.2, contribution: 5.0 },
    { feature: "is_night", value: 1, contribution: 2.6 },
    { feature: "hour", value: 3, contribution: 1.0 },
    { feature: "dist_home_km", value: 12.0, contribution: 0.5 },
  ],
  groups: [{ id: "R01_AMOUNT_VS_PATTERN", title: "Amount", contribution: 5.0 }],
  reason_codes: [
    {
      id: "R01_AMOUNT_VS_PATTERN",
      text: "Amount is 8.2 times this card's 30-day average of $64",
      share: 0.55,
      contribution: 5.0,
    },
    {
      id: "R05_TIME_AND_CATEGORY",
      text: "Online shopping purchase at night, in the 03:00 hour (UTC)",
      share: 0.4,
      contribution: 3.6,
    },
  ],
  mitigating: {
    id: "R03_TRAVEL",
    text: "12 km from home",
    share: -0.05,
    contribution: -0.5,
  },
};

export const history: CardTransaction[] = [
  {
    transaction_id: "tx-7",
    event_time: "2020-08-14T03:12:44Z",
    amount: 528.25,
    category: "shopping_net",
    decision: "block",
    risk_score: 0.97,
    degraded: false,
  },
  {
    transaction_id: "tx-6",
    event_time: "2020-08-13T18:00:00Z",
    amount: 41.0,
    category: "grocery_pos",
    decision: "approve",
    risk_score: 0.001,
    degraded: false,
  },
];

const point = (threshold: number, count: number, frauds: number, recall: number) => ({
  threshold,
  count,
  frauds,
  fraud_amount: frauds * 300,
  genuine: count - frauds,
  genuine_amount: (count - frauds) * 60,
  precision: count ? frauds / count : null,
  recall,
  dollar_recall: recall,
  alerts_per_day: count / 10,
});

export const lab: ThresholdLab = {
  model_version: "2026-11-17.1",
  rows: 10_000,
  days: 10,
  frauds: 100,
  fraud_amount: 30_000,
  costs: { review_cost: 4, margin_rate: 0.02, friction_cost: 15, review_capacity_share: 0.01 },
  points: [
    point(0, 10_000, 100, 1),
    point(0.02, 150, 90, 0.9),
    point(0.5, 60, 55, 0.55),
    point(1.01, 0, 0, 0),
  ],
  pairs: [
    [0, 0, 400, 0],
    [0, 1, 380, 0.985],
    [0, 2, 395, 0.994],
    [0, 3, 401, 1],
    [1, 1, 40, 0],
    [1, 2, 25.5, 0.009],
    [1, 3, 31, 0.015],
    [2, 2, 140, 0],
    [2, 3, 150, 0.006],
    [3, 3, 300, 0],
  ].map(([review, block, cost, share]) => ({
    review: review as number,
    block: block as number,
    cost_per_1000: cost as number,
    review_share: share as number,
    within_capacity: (share as number) <= 0.01,
  })),
  current_review: 1,
  current_block: 2,
};

export const model: ModelInfo = {
  version: "2026-11-17.1",
  manifest_sha256: "a".repeat(64),
  thresholds: { t_review: 0.02, t_block: 0.5 },
  features: ["amount", "category"],
  feature_types: ["q", "c"],
  categories: ["grocery_pos", "unknown"],
  metrics: {
    valid_early: {
      average_precision: 0.99,
      roc_auc: 0.9998,
      recall_at_1pct: 0.89,
      precision_at_1pct: 0.51,
    },
    valid_late: {
      expected_cost_per_1000: 28.4,
      decision_share: { approve: 0.99, review: 0.004, block: 0.006 },
    },
  },
};

export const monitoring: MonitoringSummary = {
  window: 200,
  model_version: "2026-11-17.1",
  decisions: { total: 200, approve: 190, review: 6, block: 4, degraded: 3 },
  alert_rate: 0.05,
  score_histogram: {
    edges: Array.from({ length: 21 }, (_, i) => i / 20),
    counts: [180, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 10],
  },
  latency_ms: { p50: 8, p95: 14, p99: 21 },
  dispositions: { fraud: 3, genuine: 1, realised_precision: 0.75 },
  drift: null,
};

export const experiments: ExperimentsFile = {
  dataset: "sparkov",
  evaluated_on: "valid_early",
  eval_rows: 194_501,
  eval_frauds: 1_151,
  seeds: [0, 1, 2],
  n_boot: 200,
  arms: [
    {
      arm: "A1",
      description: "XGBoost, no weighting",
      summary: {
        average_precision: { mean: 0.987, ci95: [0.982, 0.992], seed_std: 0.0002 },
        recall_at_1pct: { mean: 0.89, ci95: [0.86, 0.92], seed_std: 0.001 },
      },
      pr_curve: { recall: [0, 0.5, 1], precision: [1, 0.95, 0.1] },
    },
  ],
};

export const driftReport: NonNullable<MonitoringSummary["drift"]> = {
  computed_at: "2026-11-18T10:00:00Z",
  model_version: "2026-11-17.1",
  window: 10_000,
  previous_window: 10_000,
  features: [
    { feature: "amount", psi: 0.012, status: "stable", cyclical: false },
    { feature: "day_of_week", psi: 2.09, status: "alarm", cyclical: true },
  ],
  segments: [
    { category: "shopping_net", psi: 0.342, status: "alarm", n_live: 732, n_previous: 741 },
    { category: "travel", psi: null, status: "unknown", n_live: 40, n_previous: 35 },
  ],
  decision_mix: {
    psi: 0.004,
    status: "stable",
    live: { approve: 0.99, review: 0.004, block: 0.006 },
    validation: { approve: 0.99, review: 0.0035, block: 0.0059 },
  },
  alarms: [{ kind: "segment", name: "amount in shopping_net", psi: 0.342 }],
};
