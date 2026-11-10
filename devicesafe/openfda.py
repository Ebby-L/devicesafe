"""Client for the openFDA medical device adverse event (MAUDE) endpoint.

Docs: https://open.fda.gov/apis/device/event/
No key is needed for light use (1,000 requests/day). Set OPENFDA_API_KEY for more.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests

BASE = "https://api.fda.gov/device/event.json"
MAX_PER_PAGE = 1000  # openFDA hard limit per request


@dataclass
class Event:
    report_number: str
    date_received: str
    event_type: str
    brand_name: str
    generic_name: str
    manufacturer: str
    product_problems: list[str] = field(default_factory=list)
    narrative: str = ""

    @property
    def year(self) -> str:
        return self.date_received[:4]


def build_search(device: str, since: str | None = None, until: str | None = None,
                 exclude: list[str] | None = None) -> str:
    """openFDA search string for a device's generic name over a date range (YYYYMMDD).

    `exclude` drops reports whose generic name contains any of the given terms, e.g.
    ["insulin"] keeps insulin pumps out of a general infusion pump search.
    """
    term = device.strip().replace('"', "")
    q = f'device.generic_name:"{term}"'
    for x in exclude or []:
        q += f'+AND+NOT+device.generic_name:"{x.strip().replace(chr(34), "")}"'
    if since or until:
        q += f"+AND+date_received:[{since or '19900101'}+TO+{until or '20991231'}]"
    return q


def _get(params: dict, session: requests.Session | None = None) -> dict:
    key = os.getenv("OPENFDA_API_KEY")
    if key:
        params = {**params, "api_key": key}
    # openFDA expects literal '+' and ':' in search; build the query string ourselves
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    r = (session or requests).get(f"{BASE}?{qs}", timeout=30)
    if r.status_code == 404:  # openFDA's way of saying "no matches"
        return {"results": [], "meta": {"results": {"total": 0}}}
    r.raise_for_status()
    return r.json()


def parse_event(raw: dict) -> Event:
    dev = (raw.get("device") or [{}])[0]
    texts = raw.get("mdr_text") or []
    # Prefer the event description; fall back to any narrative text
    desc = [t.get("text", "") for t in texts if "event" in t.get("text_type_code", "").lower()]
    narrative = " ".join(desc or [t.get("text", "") for t in texts]).strip()
    return Event(
        report_number=raw.get("report_number", ""),
        date_received=raw.get("date_received", ""),
        event_type=raw.get("event_type") or "Unknown",
        brand_name=dev.get("brand_name", ""),
        generic_name=dev.get("generic_name", ""),
        manufacturer=dev.get("manufacturer_d_name", ""),
        product_problems=raw.get("product_problems") or [],
        narrative=narrative,
    )


def fetch_events(device: str, limit: int = 300, since: str | None = None,
                 until: str | None = None, exclude: list[str] | None = None,
                 session: requests.Session | None = None) -> list[Event]:
    """Fetch up to `limit` adverse event reports for a device, newest first."""
    search, events, skip = build_search(device, since, until, exclude), [], 0
    while len(events) < limit:
        page = min(MAX_PER_PAGE, limit - len(events))
        data = _get({"search": search, "sort": "date_received:desc",
                     "limit": page, "skip": skip}, session)
        batch = data.get("results", [])
        events += [parse_event(r) for r in batch]
        if len(batch) < page:
            break
        skip += page
    return events


def count_by(device: str, field_name: str, since: str | None = None,
             exclude: list[str] | None = None,
             session: requests.Session | None = None) -> dict[str, int]:
    """Server-side counts across ALL matching reports (not just the fetched sample)."""
    data = _get({"search": build_search(device, since, exclude=exclude), "count": field_name}, session)
    return {r["term"]: r["count"] for r in data.get("results", [])}
