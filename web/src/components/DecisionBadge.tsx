// A decision shown with an icon and a word, never by colour alone.
import type { Decision } from "../api/client";

const LOOK: Record<Decision, { icon: string; label: string }> = {
  approve: { icon: "✓", label: "Approve" },
  review: { icon: "●", label: "Review" },
  block: { icon: "■", label: "Block" },
};

export function DecisionBadge({ decision }: { decision: Decision }) {
  const { icon, label } = LOOK[decision];
  return (
    <span className={`badge badge-${decision}`}>
      <span aria-hidden="true">{icon}</span> {label}
    </span>
  );
}
