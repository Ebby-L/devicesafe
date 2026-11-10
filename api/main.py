"""DeviceSafe API.  Run: uvicorn api.main:app --reload   Docs: http://127.0.0.1:8000/docs"""
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from devicesafe import openfda
from devicesafe.retrieval import event_type_counts, problem_frequencies
from devicesafe.review import review

app = FastAPI(title="DeviceSafe API", version="1.0.0",
              description="Grounded design-risk review of FDA medical device adverse event reports.")


@lru_cache(maxsize=32)
def _events(device: str, limit: int, since: str | None) -> tuple:
    try:
        return tuple(openfda.fetch_events(device, limit=limit, since=since))
    except Exception as exc:  # network or openFDA error
        raise HTTPException(502, f"openFDA request failed: {exc}") from exc


@app.get("/events")
def events(device: str = Query(..., examples=["infusion pump"]),
           limit: int = Query(200, ge=1, le=2000),
           since: str | None = Query(None, pattern=r"^\d{8}$", description="YYYYMMDD")):
    evs = list(_events(device.lower(), limit, since))
    return {"device": device, "fetched": len(evs),
            "event_types": event_type_counts(evs),
            "top_problems": problem_frequencies(evs, 10),
            "reports": [{"report_number": e.report_number, "date": e.date_received,
                         "event_type": e.event_type, "brand": e.brand_name,
                         "narrative": e.narrative[:400]} for e in evs[:50]]}


class ReviewRequest(BaseModel):
    device: str = Field(..., examples=["infusion pump"])
    question: str | None = Field(None, examples=["What failure modes lead to over-infusion?"])
    limit: int = Field(300, ge=10, le=2000)
    since: str | None = Field(None, pattern=r"^\d{8}$")


@app.post("/review")
def run_review(req: ReviewRequest):
    evs = list(_events(req.device.lower(), req.limit, req.since))
    if not evs:
        raise HTTPException(404, f"No adverse event reports found for '{req.device}'.")
    try:
        return {"device": req.device, **review(req.device, evs, req.question)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
