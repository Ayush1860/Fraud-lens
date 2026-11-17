// Reason codes in words, first on the case view.
import type { ReasonCode } from "../api/client";
import { percent } from "../lib/format";

export function ReasonList({ reasons }: { reasons: ReasonCode[] }) {
  if (reasons.length === 0) {
    return <p className="muted">No reason could be worded for this decision.</p>;
  }
  return (
    <ol className="reasons">
      {reasons.map((reason) => (
        <li key={reason.id}>
          <span className="reason-text">{reason.text}</span>
          <span className="reason-share">
            {reason.contribution === 0 && reason.share === 0
              ? "rule"
              : `${percent(reason.share, 0)} of the push`}
          </span>
        </li>
      ))}
    </ol>
  );
}
