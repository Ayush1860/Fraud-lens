// Screen 1: open alerts by priority (risk score times amount), polled every 5 seconds.
import { useState } from "react";
import { Link } from "react-router";

import { QUEUE_POLL_MS, useAlerts, type AlertFilter } from "../api/queries";
import { DecisionBadge } from "../components/DecisionBadge";
import { Empty, Loaded } from "../components/Status";
import { age, money, score } from "../lib/format";

export function AlertQueue() {
  const [decision, setDecision] = useState<AlertFilter["decision"]>(undefined);
  const alerts = useAlerts({ status: "open", decision });

  return (
    <section aria-labelledby="queue-title">
      <header className="page-header">
        <div>
          <h1 id="queue-title">Alert queue</h1>
          <p className="muted">
            Open alerts, highest dollars at risk first. Refreshes every {QUEUE_POLL_MS / 1000}{" "}
            seconds.
          </p>
        </div>
        <label className="field">
          <span>Decision</span>
          <select
            value={decision ?? ""}
            onChange={(event) =>
              setDecision((event.target.value || undefined) as AlertFilter["decision"])
            }
          >
            <option value="">Review and block</option>
            <option value="review">Review only</option>
            <option value="block">Block only</option>
          </select>
        </label>
      </header>

      <Loaded query={alerts} what="alerts">
        {(page) =>
          page.items.length === 0 ? (
            <Empty>No open alerts. New ones appear here as transactions are flagged.</Empty>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Decision</th>
                  <th scope="col" className="num">
                    At risk
                  </th>
                  <th scope="col" className="num">
                    Amount
                  </th>
                  <th scope="col" className="num">
                    Score
                  </th>
                  <th scope="col">Category</th>
                  <th scope="col">Top reason</th>
                  <th scope="col" className="num">
                    Age
                  </th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((alert) => (
                  <tr key={alert.id}>
                    <td>
                      <DecisionBadge decision={alert.decision} />
                    </td>
                    <td className="num">{money(alert.priority)}</td>
                    <td className="num">
                      <Link
                        className="row-link"
                        to={`/cases/${alert.transaction_id}?alert=${alert.id}`}
                      >
                        {money(alert.amount)}
                      </Link>
                    </td>
                    <td className="num">{score(alert.risk_score)}</td>
                    <td>{alert.category}</td>
                    <td>{alert.top_reason ?? <span className="muted">none worded</span>}</td>
                    <td className="num">{age(alert.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        }
      </Loaded>
    </section>
  );
}
