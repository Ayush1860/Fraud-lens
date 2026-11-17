// Screen 4: the released model and how the service is behaving. Read-only.
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts";

import type { ModelInfo, MonitoringSummary } from "../api/client";
import { useModel, useMonitoring } from "../api/queries";
import { Empty, Loaded } from "../components/Status";
import { numberAt } from "../lib/experiments";
import { count, money, percent } from "../lib/format";

export function ModelPage() {
  const model = useModel();
  const monitoring = useMonitoring();
  return (
    <section aria-labelledby="model-title">
      <header className="page-header">
        <div>
          <h1 id="model-title">Model and monitoring</h1>
          <p className="muted">The released model, and the service's recent decisions.</p>
        </div>
      </header>
      <Loaded query={model} what="the model">
        {(info) => (
          <>
            <ModelCard info={info} />
            <h2>Recent decisions</h2>
            <Loaded query={monitoring} what="monitoring">
              {(summary) => <Monitoring summary={summary} info={info} />}
            </Loaded>
          </>
        )}
      </Loaded>
    </section>
  );
}

function ModelCard({ info }: { info: ModelInfo }) {
  const m = info.metrics;
  const rows: [string, string][] = [
    ["Version", info.version],
    ["Manifest SHA-256", `${info.manifest_sha256.slice(0, 16)}…`],
    ["Review threshold", info.thresholds.t_review.toFixed(4)],
    ["Block threshold", info.thresholds.t_block.toFixed(4)],
    ["Features", String(info.features.length)],
    ["Average precision (validation)", fixed(numberAt(m, "valid_early", "average_precision"))],
    ["ROC AUC (validation)", fixed(numberAt(m, "valid_early", "roc_auc"))],
    ["Recall at a 1% review budget", percent(numberAt(m, "valid_early", "recall_at_1pct"))],
    ["Precision at a 1% review budget", percent(numberAt(m, "valid_early", "precision_at_1pct"))],
    [
      "Expected cost per 1,000 (validation)",
      moneyOrNa(numberAt(m, "valid_late", "expected_cost_per_1000")),
    ],
  ];
  return (
    <dl className="card-grid">
      {rows.map(([name, value]) => (
        <div key={name}>
          <dt>{name}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Monitoring({ summary, info }: { summary: MonitoringSummary; info: ModelInfo }) {
  if (summary.window === 0) {
    return <Empty>No decisions yet. Run the replay client to send traffic.</Empty>;
  }
  const kinds = ["approve", "review", "block"] as const;
  const histogram = summary.score_histogram.counts.map((n, i) => ({
    bin: `${(summary.score_histogram.edges[i] ?? 0).toFixed(2)}`,
    // Empty bins are left blank: a log scale has no zero.
    count: n > 0 ? n : null,
  }));
  const { latency_ms: latency, dispositions } = summary;
  return (
    <>
      <div className="two-up">
        <table className="table" aria-label="Decision mix">
          <thead>
            <tr>
              <th scope="col">Decision</th>
              <th scope="col" className="num">
                Last {count(summary.window)}
              </th>
              <th scope="col" className="num">
                Validation
              </th>
            </tr>
          </thead>
          <tbody>
            {kinds.map((kind) => (
              <tr key={kind}>
                <th scope="row">{kind}</th>
                <td className="num">{percent(summary.decisions[kind] / summary.window, 2)}</td>
                <td className="num">
                  {percent(numberAt(info.metrics, "valid_late", "decision_share", kind), 2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <dl className="card-grid compact">
          <div>
            <dt>Alert rate</dt>
            <dd>{percent(summary.alert_rate, 2)}</dd>
          </div>
          <div>
            <dt>Scored without history</dt>
            <dd>{count(summary.decisions.degraded)}</dd>
          </div>
          <div>
            <dt>Decision latency p50 / p95 / p99</dt>
            <dd>
              {[latency.p50, latency.p95, latency.p99]
                .map((v) => (v === null ? "n/a" : `${v.toFixed(0)} ms`))
                .join(" / ")}
            </dd>
          </div>
          <div>
            <dt>Realised precision of labelled alerts</dt>
            <dd>
              {percent(dispositions.realised_precision)} ({count(dispositions.fraud)} fraud,{" "}
              {count(dispositions.genuine)} genuine)
            </dd>
          </div>
        </dl>
      </div>
      <figure>
        <BarChart width={640} height={220} data={histogram} accessibilityLayer>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="bin" interval={3} />
          <YAxis scale="log" domain={[0.8, "auto"]} allowDataOverflow allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="var(--ink)" isAnimationActive={false} />
        </BarChart>
        <figcaption>
          Risk scores of the recent decisions, in 20 bins from 0 to 1. Counts on a log scale: almost
          every score is near 0, and the tail is what matters.
        </figcaption>
      </figure>
      <h2>Feature drift</h2>
      {summary.drift ? (
        <table className="table" aria-label="Drift">
          <tbody>
            {Object.entries(summary.drift).map(([feature, psi]) => (
              <tr key={feature}>
                <th scope="row">{feature}</th>
                <td className="num">{psi.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Empty>Population stability per feature arrives with the drift job in phase 6.</Empty>
      )}
    </>
  );
}

function fixed(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(4);
}

function moneyOrNa(value: number | null): string {
  return value === null ? "n/a" : money(value);
}
