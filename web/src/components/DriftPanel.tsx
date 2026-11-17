// The drift report: an alarm banner when anything is over the line, then the three
// comparisons behind it. PSI under 0.1 reads as stable, over 0.25 as an alarm.
import type { Schemas } from "../api/client";
import { count, percent, utc } from "../lib/format";
import { Empty } from "./Status";

type Drift = Schemas["DriftReport"];
type DriftStatus = Schemas["FeatureDrift"]["status"];

const LOOK: Record<DriftStatus, { icon: string; label: string }> = {
  stable: { icon: "✓", label: "Stable" },
  watch: { icon: "●", label: "Watch" },
  alarm: { icon: "▲", label: "Alarm" },
  unknown: { icon: "–", label: "Too few rows" },
};

function StatusBadge({ status }: { status: DriftStatus }) {
  const { icon, label } = LOOK[status];
  return (
    <span className={`badge drift-${status}`}>
      <span aria-hidden="true">{icon}</span> {label}
    </span>
  );
}

function psiText(psi: number | null): string {
  return psi === null ? "n/a" : psi.toFixed(3);
}

export function DriftPanel({ drift }: { drift: Drift | null }) {
  if (drift === null) {
    return <Empty>The first drift report is computed shortly after the service starts.</Empty>;
  }
  if (drift.window === 0) {
    return <Empty>No decisions yet, so nothing to compare.</Empty>;
  }
  const features = [...drift.features].sort((a, b) => (b.psi ?? -1) - (a.psi ?? -1));
  return (
    <div className="drift">
      {drift.alarms.length > 0 ? (
        <div className="alarm" role="status" aria-live="polite">
          <strong>Drift alarm.</strong>{" "}
          {drift.alarms.map((a) => `${a.name} (PSI ${a.psi.toFixed(2)})`).join("; ")}. See the
          runbook before changing anything.
        </div>
      ) : (
        <p className="muted small">No alarms in the latest report.</p>
      )}
      <p className="muted small">
        Computed {utc(drift.computed_at)} over the newest {count(drift.window)} decisions; the{" "}
        {count(drift.previous_window)} before them are the baseline for the per-category check.
      </p>

      <div className="two-up">
        <table className="table" aria-label="Amount by category">
          <caption>Amount within each category, against the previous window</caption>
          <thead>
            <tr>
              <th scope="col">Category</th>
              <th scope="col" className="num">
                PSI
              </th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {drift.segments.map((s) => (
              <tr key={s.category}>
                <th scope="row">{s.category}</th>
                <td className="num">{psiText(s.psi)}</td>
                <td>
                  <StatusBadge status={s.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <table className="table" aria-label="Decision mix drift">
          <caption>Decision mix against validation</caption>
          <thead>
            <tr>
              <th scope="col">Decision</th>
              <th scope="col" className="num">
                Live
              </th>
              <th scope="col" className="num">
                Validation
              </th>
            </tr>
          </thead>
          <tbody>
            {(["approve", "review", "block"] as const).map((kind) => (
              <tr key={kind}>
                <th scope="row">{kind}</th>
                <td className="num">{percent(drift.decision_mix.live[kind] ?? null, 2)}</td>
                <td className="num">{percent(drift.decision_mix.validation[kind] ?? null, 2)}</td>
              </tr>
            ))}
            <tr>
              <th scope="row">PSI</th>
              <td className="num">{psiText(drift.decision_mix.psi)}</td>
              <td>
                <StatusBadge status={drift.decision_mix.status} />
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <table className="table" aria-label="Features against training">
        <caption>Each feature against its training histogram</caption>
        <thead>
          <tr>
            <th scope="col">Feature</th>
            <th scope="col" className="num">
              PSI
            </th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {features.map((f) => (
            <tr key={f.feature}>
              <th scope="row">{f.feature}</th>
              <td className="num">{psiText(f.psi)}</td>
              <td>
                <StatusBadge status={f.status} />
                {f.cyclical ? (
                  <span className="muted small"> calendar: the window is shorter than a week</span>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
