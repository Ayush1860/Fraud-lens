// The contribution waterfall: from the model's baseline to this transaction's margin,
// in log-odds. It supports the reason codes; it does not replace them.
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";

import { featureLabel, logOdds } from "../lib/format";
import { buildWaterfall, type Contribution, type WaterfallStep } from "../lib/waterfall";

const FILL: Record<WaterfallStep["kind"], string> = {
  baseline: "var(--muted-bar)",
  up: "var(--risk)",
  down: "var(--safe)",
  total: "var(--ink)",
};

export function Waterfall({
  bias,
  contributions,
}: {
  bias: number;
  contributions: Contribution[];
}) {
  const steps = buildWaterfall(bias, contributions).map((step) => ({
    ...step,
    name: step.kind === "up" || step.kind === "down" ? featureLabel(step.label) : step.label,
    range: [Math.min(step.start, step.end), Math.max(step.start, step.end)] as [number, number],
  }));
  return (
    <figure className="waterfall">
      <BarChart
        width={640}
        height={28 * steps.length + 40}
        data={steps}
        layout="vertical"
        margin={{ left: 8, right: 24, top: 8, bottom: 8 }}
        accessibilityLayer
      >
        <CartesianGrid horizontal={false} stroke="var(--grid)" />
        <XAxis
          type="number"
          domain={["auto", "auto"]}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <YAxis type="category" dataKey="name" width={200} interval={0} />
        <ReferenceLine x={0} stroke="var(--grid-strong)" />
        <Tooltip
          formatter={(_value, _name, item) => logOdds((item.payload as WaterfallStep).value)}
        />
        <Bar dataKey="range" isAnimationActive={false}>
          {steps.map((step) => (
            <Cell key={step.label} fill={FILL[step.kind]} />
          ))}
        </Bar>
      </BarChart>
      <figcaption>
        Contributions in log-odds. Red pushes towards fraud, green away from it; the last bar is the
        model's output for this transaction.
      </figcaption>
      <table className="sr-only">
        <caption>Waterfall values</caption>
        <tbody>
          {steps.map((step) => (
            <tr key={step.label}>
              <th scope="row">{step.name}</th>
              <td>{logOdds(step.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
