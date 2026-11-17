// Screen 5: the imbalance study, read from docs/results/experiments.json. Read-only.
import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";

import { useExperiments } from "../api/queries";
import { Empty, Loaded } from "../components/Status";
import { curvePoints, type Arm, type ExperimentsFile } from "../lib/experiments";
import { count, percent } from "../lib/format";

const METRICS: [key: string, label: string, asPercent: boolean][] = [
  ["average_precision", "Average precision", false],
  ["recall_at_1pct", "Recall at 1%", true],
  ["precision_at_1pct", "Precision at 1%", true],
  ["dollar_recall_at_1pct", "Fraud $ at 1%", true],
  ["brier", "Brier", false],
];

const SERIES = ["#1f2933", "#b42318", "#1769aa", "#2e7d32", "#8a5a00", "#6b3fa0", "#5f6b76"];

export function Experiments() {
  const experiments = useExperiments();
  return (
    <section aria-labelledby="experiments-title">
      <header className="page-header">
        <div>
          <h1 id="experiments-title">Experiments</h1>
          <p className="muted">
            Seven ways of handling the class imbalance, each run with several seeds. Intervals are
            95% day-block bootstrap intervals.
          </p>
        </div>
      </header>
      <Loaded query={experiments} what="the experiments">
        {(data) =>
          data.arms.length === 0 ? <Empty>No experiment results yet.</Empty> : <Study data={data} />
        }
      </Loaded>
    </section>
  );
}

function Study({ data }: { data: ExperimentsFile }) {
  return (
    <>
      <p className="muted small">
        Evaluated on the {data.evaluated_on.replace("_", " ")} slice: {count(data.eval_rows)}{" "}
        transactions, {count(data.eval_frauds)} frauds; seeds {data.seeds.join(", ")}.
      </p>
      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Arm</th>
              {METRICS.map(([key, label]) => (
                <th scope="col" key={key} className="num">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.arms.map((arm) => (
              <tr key={arm.arm}>
                <th scope="row">
                  {arm.arm} <span className="muted">{arm.description}</span>
                </th>
                {METRICS.map(([key, , asPercent]) => (
                  <td key={key} className="num">
                    <Interval arm={arm} metric={key} asPercent={asPercent} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figure>
        <LineChart
          width={640}
          height={340}
          margin={{ left: 8, right: 16, bottom: 24 }}
          accessibilityLayer
        >
          <CartesianGrid stroke="var(--grid)" />
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => percent(v, 0)}
            label={{ value: "Recall", position: "insideBottom", offset: -12 }}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => percent(v, 0)}
            label={{ value: "Precision", angle: -90, position: "insideLeft" }}
          />
          <Tooltip formatter={(v) => percent(Number(v))} />
          <Legend verticalAlign="top" />
          {data.arms.map((arm, i) => (
            <Line
              key={arm.arm}
              data={curvePoints(arm)}
              dataKey="precision"
              name={arm.arm}
              stroke={SERIES[i % SERIES.length]}
              strokeDasharray={i % 2 ? "4 3" : undefined}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
        <figcaption>
          Precision-recall curves, one per arm: interpolated precision for the first seed.
        </figcaption>
      </figure>
    </>
  );
}

function Interval({ arm, metric, asPercent }: { arm: Arm; metric: string; asPercent: boolean }) {
  const value = arm.summary[metric];
  if (!value) return <span className="muted">n/a</span>;
  const show = (v: number) => (asPercent ? percent(v) : v.toFixed(4));
  return (
    <>
      {show(value.mean)}
      <span className="interval">
        [{show(value.ci95[0])}, {show(value.ci95[1])}]
      </span>
    </>
  );
}
