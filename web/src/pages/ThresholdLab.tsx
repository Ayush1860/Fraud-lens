// Screen 3: what other thresholds would have done on the validation slice. Read-only.
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceDot, Tooltip, XAxis, YAxis } from "recharts";

import type { ThresholdLab as Lab } from "../api/client";
import { useThresholdLab } from "../api/queries";
import { Loaded } from "../components/Status";
import { count, money, percent } from "../lib/format";
import { operatingPoint, thresholdLabel, type OperatingPoint } from "../lib/thresholds";

export function ThresholdLab() {
  const lab = useThresholdLab();
  return (
    <section aria-labelledby="lab-title">
      <header className="page-header">
        <div>
          <h1 id="lab-title">Threshold lab</h1>
          <p className="muted">
            A simulation on the validation slice the thresholds were chosen on. Changing a slider
            here changes nothing in the service.
          </p>
        </div>
      </header>
      <Loaded query={lab} what="the threshold curve">
        {(data) => <Simulator lab={data} />}
      </Loaded>
    </section>
  );
}

function Simulator({ lab }: { lab: Lab }) {
  const [review, setReview] = useState(lab.current_review);
  const [block, setBlock] = useState(lab.current_block);
  const last = lab.points.length - 1;
  const point = operatingPoint(lab, review, Math.max(block, review));
  const released = operatingPoint(lab, lab.current_review, lab.current_block);

  return (
    <div className="lab">
      <div className="sliders">
        <Slider
          id="review"
          label="Review threshold"
          index={review}
          max={last}
          lab={lab}
          onChange={(value) => {
            setReview(value);
            if (block < value) setBlock(value);
          }}
        />
        <Slider
          id="block"
          label="Block threshold"
          index={Math.max(block, review)}
          min={review}
          max={last}
          lab={lab}
          onChange={setBlock}
        />
        <button
          type="button"
          className="button button-quiet"
          disabled={point.isReleased}
          onClick={() => {
            setReview(lab.current_review);
            setBlock(lab.current_block);
          }}
        >
          Back to the released thresholds
        </button>
        <p className="muted small">
          {lab.points.length} possible values: the calibrated model gives exactly this many distinct
          scores. Validation slice: {count(lab.rows)} transactions over {lab.days} days.
        </p>
      </div>

      <Stats point={point} released={released} capacity={lab.costs.review_capacity_share} />

      <figure>
        <LineChart
          width={520}
          height={300}
          data={lab.points.filter((p) => p.precision !== null && p.recall !== null)}
          margin={{ left: 8, right: 16, top: 8, bottom: 24 }}
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
          <Line dataKey="precision" stroke="var(--ink)" dot={{ r: 2 }} isAnimationActive={false} />
          {released.recall !== null && released.precision !== null ? (
            <ReferenceDot
              x={released.recall}
              y={released.precision}
              r={7}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={2}
              label={{ value: "released", position: "left", offset: 12 }}
            />
          ) : null}
          {point.recall !== null && point.precision !== null ? (
            <ReferenceDot x={point.recall} y={point.precision} r={5} fill="var(--risk)" />
          ) : null}
        </LineChart>
        <figcaption>
          Precision against recall at each review threshold. The ring is the released point; the dot
          is the one selected.
        </figcaption>
      </figure>
    </div>
  );
}

function Slider({
  id,
  label,
  index,
  min = 0,
  max,
  lab,
  onChange,
}: {
  id: string;
  label: string;
  index: number;
  min?: number;
  max: number;
  lab: Lab;
  onChange: (index: number) => void;
}) {
  const threshold = lab.points[index]?.threshold ?? Number.NaN;
  return (
    <label className="field" htmlFor={id}>
      <span>
        {label}: <strong>{thresholdLabel(threshold)}</strong>
      </span>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={1}
        value={index}
        aria-valuetext={thresholdLabel(threshold)}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function Stats({
  point,
  released,
  capacity,
}: {
  point: OperatingPoint;
  released: OperatingPoint;
  capacity: number;
}) {
  const rows: [string, string, string][] = [
    ["Expected cost per 1,000 transactions", money(point.costPer1000), money(released.costPer1000)],
    ["Alerts per day (review and block)", count(point.alertsPerDay), count(released.alertsPerDay)],
    ["Blocks per day", count(point.blocksPerDay), count(released.blocksPerDay)],
    ["Share sent to review", percent(point.reviewShare, 2), percent(released.reviewShare, 2)],
    ["Precision of alerts", percent(point.precision), percent(released.precision)],
    ["Frauds caught (recall)", percent(point.recall), percent(released.recall)],
    ["Fraud dollars caught", percent(point.dollarRecall), percent(released.dollarRecall)],
  ];
  return (
    <div>
      <table className="table stats" aria-label="Operating point">
        <thead>
          <tr>
            <th scope="col">On the validation slice</th>
            <th scope="col" className="num">
              Selected
            </th>
            <th scope="col" className="num">
              Released
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, selected, current]) => (
            <tr key={name}>
              <th scope="row">{name}</th>
              <td className="num">{selected}</td>
              <td className="num">{current}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className={point.withinCapacity ? "muted small" : "warning small"} role="status">
        {point.isReleased
          ? "These are the released thresholds."
          : point.withinCapacity
            ? `Within the review capacity of ${percent(capacity, 1)} of transactions.`
            : `Over the review capacity of ${percent(capacity, 1)} of transactions.`}
      </p>
    </div>
  );
}
