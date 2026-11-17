"""Audit schema: decisions, alerts, dispositions and model versions.

Revision ID: 0001
Revises:
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APPEND_ONLY = ("decisions", "dispositions", "model_versions")


def upgrade() -> None:
    """Create the tables, their indexes and the append-only triggers."""
    op.create_table(
        "decisions",
        sa.Column("transaction_id", sa.Text, primary_key=True),
        sa.Column("card_id", sa.Text, nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("amount", sa.Float, nullable=False),
        sa.Column("category", sa.Text, nullable=False),
        sa.Column("features", JSONB, nullable=False),
        sa.Column("margin", sa.Float, nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("t_review", sa.Float, nullable=False),
        sa.Column("t_block", sa.Float, nullable=False),
        sa.Column("model_version", sa.Text, nullable=False),
        sa.Column("reason_codes", JSONB, nullable=False),
        sa.Column("degraded", sa.Boolean, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        # clock_timestamp(), not now(): rows inserted in one transaction stay ordered.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
        sa.CheckConstraint("decision IN ('approve', 'review', 'block')", name="decision_kind"),
        sa.CheckConstraint("amount > 0", name="amount_positive"),
        sa.CheckConstraint("score >= 0 AND score <= 1", name="score_probability"),
    )
    op.create_index("decisions_created_at", "decisions", ["created_at"])
    op.create_index("decisions_card_event", "decisions", ["card_id", "event_time"])

    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column(
            "transaction_id",
            sa.Text,
            sa.ForeignKey("decisions.transaction_id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("priority", sa.Float, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="open"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
        sa.CheckConstraint("status IN ('open', 'closed')", name="alert_status"),
    )
    op.create_index("alerts_queue", "alerts", ["status", sa.text("priority DESC"), "id"])

    op.create_table(
        "dispositions",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("alert_id", sa.BigInteger, sa.ForeignKey("alerts.id"), nullable=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("analyst", sa.Text, nullable=False),
        sa.Column("note", sa.Text),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
        sa.CheckConstraint("label IN ('fraud', 'genuine')", name="disposition_label"),
    )
    op.create_index("dispositions_alert", "dispositions", ["alert_id"])

    op.create_table(
        "model_versions",
        sa.Column("version", sa.Text, primary_key=True),
        sa.Column("manifest_hash", sa.Text, nullable=False),
        sa.Column("metrics", JSONB, nullable=False),
        sa.Column(
            "activated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
    )

    op.execute(
        """
        CREATE FUNCTION fraudlens_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME;
        END;
        $$
        """
    )
    for table in APPEND_ONLY:
        op.execute(
            f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION fraudlens_append_only()"
        )

    op.execute(
        """
        CREATE FUNCTION fraudlens_alert_status_only() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'table alerts is append-only';
            END IF;
            IF (NEW.id, NEW.transaction_id, NEW.priority, NEW.created_at)
               IS DISTINCT FROM (OLD.id, OLD.transaction_id, OLD.priority, OLD.created_at) THEN
                RAISE EXCEPTION 'only alerts.status may change';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER alerts_status_only BEFORE UPDATE OR DELETE ON alerts "
        "FOR EACH ROW EXECUTE FUNCTION fraudlens_alert_status_only()"
    )
    for table in (*APPEND_ONLY, "alerts"):
        op.execute(f"REVOKE UPDATE, DELETE, TRUNCATE ON {table} FROM PUBLIC")


def downgrade() -> None:
    """Drop everything ``upgrade`` created."""
    for table in ("dispositions", "alerts", "model_versions", "decisions"):
        op.drop_table(table)
    op.execute("DROP FUNCTION IF EXISTS fraudlens_append_only()")
    op.execute("DROP FUNCTION IF EXISTS fraudlens_alert_status_only()")
