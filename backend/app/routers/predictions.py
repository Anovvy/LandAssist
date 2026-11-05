"""Prediction lifecycle state management endpoints for LandAssist."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.db import db

router = APIRouter(prefix="/v1/predictions", tags=["predictions"])


class BatchCommitItem(BaseModel):
    id: int
    osm_id: int | None = None


class BatchCommitRequest(BaseModel):
    items: list[BatchCommitItem]
    reviewed_by: str | None = Field(default=None, description="OSM username or editor ID")


@router.post("/{prediction_id}/stage")
async def stage_prediction(prediction_id: int, user: str | None = Query(default=None)) -> dict[str, Any]:
    """Mark a prediction as staged (user clicked Space / Add to features)."""
    success = await db.update_prediction_status(prediction_id, "staged", reviewed_by=user)
    if not success:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
    return {"status": "ok", "prediction_id": prediction_id, "lifecycle": "staged"}


@router.post("/{prediction_id}/revert")
async def revert_prediction(prediction_id: int) -> dict[str, Any]:
    """Revert a prediction back to unreviewed (user pressed Ctrl+Z / Undo)."""
    success = await db.update_prediction_status(prediction_id, "unreviewed")
    if not success:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
    return {"status": "ok", "prediction_id": prediction_id, "lifecycle": "unreviewed"}


@router.post("/{prediction_id}/reject")
async def reject_prediction(prediction_id: int, user: str | None = Query(default=None)) -> dict[str, Any]:
    """Mark a prediction as rejected (user clicked Ignore / Delete)."""
    success = await db.update_prediction_status(prediction_id, "rejected", reviewed_by=user)
    if not success:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
    return {"status": "ok", "prediction_id": prediction_id, "lifecycle": "rejected"}


@router.post("/batch-commit")
async def batch_commit_predictions(payload: BatchCommitRequest) -> dict[str, Any]:
    """Mark predictions as accepted after successful OpenStreetMap changeset upload."""
    raw_items = [item.model_dump() for item in payload.items]
    updated_count = await db.batch_commit_predictions(raw_items, reviewed_by=payload.reviewed_by)
    return {"status": "ok", "committed_count": updated_count}
