"""The alert queue and analyst dispositions."""

from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from fraudlens.service.schemas import AlertPage, DispositionIn, DispositionOut
from fraudlens.service.state import ServiceState, get_state, require_api_key

router = APIRouter(prefix="/v1/alerts", dependencies=[Depends(require_api_key)])

State = Annotated[ServiceState, Depends(get_state)]


@router.get("", response_model=AlertPage)
async def list_alerts(
    state: State,
    status_filter: Annotated[Literal["open", "closed"] | None, Query(alias="status")] = None,
    decision: Annotated[Literal["review", "block"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
) -> Any:
    """Alerts ordered by priority (risk score times amount), highest first.

    Pass the returned ``next_cursor`` back as ``cursor`` for the next page.
    """
    try:
        rows, next_cursor = await state.audit.list_alerts(
            status=status_filter, decision=decision, limit=limit, cursor=cursor
        )
    except ValueError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    return {"items": [asdict(row) for row in rows], "next_cursor": next_cursor}


@router.post(
    "/{alert_id}/disposition",
    response_model=DispositionOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_disposition(
    alert_id: Annotated[int, Path(ge=1)], body: DispositionIn, state: State
) -> Any:
    """Record an analyst's label and close the alert. Labels are append-only."""
    row = await state.audit.add_disposition(
        alert_id, label=body.label, analyst=body.analyst, note=body.note
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such alert")
    return asdict(row)
