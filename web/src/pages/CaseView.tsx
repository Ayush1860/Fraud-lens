// Screen 2: why one transaction was flagged, and the analyst's verdict.
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import type { CardTransaction, DecisionRecord, Explanation } from "../api/client";
import { useCardHistory, useDecision, useDisposition, useExplanation } from "../api/queries";
import { DecisionBadge } from "../components/DecisionBadge";
import { ReasonList } from "../components/Reasons";
import { Empty, ErrorState, Loaded } from "../components/Status";
import { Waterfall } from "../components/Waterfall";
import { money, score, utc } from "../lib/format";

export function CaseView() {
  const { transactionId = "" } = useParams();
  const [search] = useSearchParams();
  const alertId = Number(search.get("alert")) || null;
  const decision = useDecision(transactionId);
  const explanation = useExplanation(transactionId);
  const history = useCardHistory(transactionId);

  return (
    <section aria-labelledby="case-title">
      <p>
        <Link to="/">← Alert queue</Link>
      </p>
      <Loaded query={decision} what="the decision">
        {(record) => (
          <>
            <Facts record={record} />
            <div className="case-grid">
              <div>
                <h2>Why it was flagged</h2>
                <Loaded query={explanation} what="the explanation">
                  {(e) => <Why explanation={e} />}
                </Loaded>
              </div>
              <div>
                <h2>Verdict</h2>
                {alertId === null ? (
                  <Empty>Open this case from the alert queue to record a verdict.</Empty>
                ) : (
                  <DispositionForm alertId={alertId} />
                )}
              </div>
            </div>
            <h2>This card's last 20 transactions</h2>
            <Loaded query={history} what="the card's history">
              {(rows) => <History rows={rows} current={record.transaction_id} />}
            </Loaded>
          </>
        )}
      </Loaded>
    </section>
  );
}

function Facts({ record }: { record: DecisionRecord }) {
  return (
    <header className="page-header">
      <div>
        <h1 id="case-title">
          {money(record.amount)} · {record.category}
        </h1>
        <p className="muted">
          {utc(record.event_time)} · card {record.card_id.slice(0, 8)}… · model{" "}
          {record.model_version}
        </p>
      </div>
      <dl className="facts">
        <div>
          <dt>Decision</dt>
          <dd>
            <DecisionBadge decision={record.decision} />
          </dd>
        </div>
        <div>
          <dt>Risk score</dt>
          <dd>{score(record.risk_score)}</dd>
        </div>
        <div>
          <dt>Thresholds</dt>
          <dd>
            review {record.t_review.toFixed(3)} · block {record.t_block.toFixed(3)}
          </dd>
        </div>
        {record.degraded ? (
          <div>
            <dt>History</dt>
            <dd className="warning">Unavailable: scored on this transaction alone</dd>
          </div>
        ) : null}
      </dl>
    </header>
  );
}

function Why({ explanation }: { explanation: Explanation }) {
  return (
    <>
      <ReasonList reasons={explanation.reason_codes} />
      {explanation.mitigating ? (
        <p className="mitigating">
          <strong>Pulling the other way:</strong> {explanation.mitigating.text}
        </p>
      ) : null}
      <Waterfall bias={explanation.bias} contributions={explanation.contributions} />
    </>
  );
}

function DispositionForm({ alertId }: { alertId: number }) {
  const [analyst, setAnalyst] = useState("analyst");
  const [note, setNote] = useState("");
  const disposition = useDisposition();

  if (disposition.isSuccess) {
    const label = disposition.data.label === "fraud" ? "confirmed fraud" : "marked genuine";
    return (
      <p className="status status-done" role="status">
        Recorded: {label} by {disposition.data.analyst}. The alert is closed.
      </p>
    );
  }
  const submit = (label: "fraud" | "genuine") =>
    disposition.mutate({ alertId, label, analyst, note });

  return (
    <form className="verdict" onSubmit={(event) => event.preventDefault()}>
      <label className="field">
        <span>Analyst</span>
        <input
          value={analyst}
          onChange={(e) => setAnalyst(e.target.value)}
          required
          maxLength={64}
        />
      </label>
      <label className="field">
        <span>Note</span>
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={3}
          maxLength={2000}
          placeholder="What you checked, for the next person"
        />
      </label>
      <div className="actions">
        <button
          type="button"
          className="button button-risk"
          disabled={disposition.isPending || !analyst.trim()}
          onClick={() => submit("fraud")}
        >
          Confirm fraud
        </button>
        <button
          type="button"
          className="button"
          disabled={disposition.isPending || !analyst.trim()}
          onClick={() => submit("genuine")}
        >
          Mark genuine
        </button>
      </div>
      {disposition.isError ? <ErrorState error={disposition.error} /> : null}
    </form>
  );
}

function History({ rows, current }: { rows: CardTransaction[]; current: string }) {
  if (rows.length === 0) return <Empty>No earlier transactions recorded for this card.</Empty>;
  return (
    <table className="table">
      <thead>
        <tr>
          <th scope="col">When</th>
          <th scope="col" className="num">
            Amount
          </th>
          <th scope="col">Category</th>
          <th scope="col">Decision</th>
          <th scope="col" className="num">
            Score
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr
            key={row.transaction_id}
            className={row.transaction_id === current ? "current" : undefined}
            aria-current={row.transaction_id === current ? "true" : undefined}
          >
            <td>
              {utc(row.event_time)}
              {row.transaction_id === current ? <strong> · this one</strong> : null}
            </td>
            <td className="num">{money(row.amount)}</td>
            <td>{row.category}</td>
            <td>
              <DecisionBadge decision={row.decision} />
            </td>
            <td className="num">
              {score(row.risk_score)}
              {row.degraded ? <span title="Scored without history"> *</span> : null}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
