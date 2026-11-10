"""Offline tests: openFDA responses and the LLM are both faked."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from devicesafe import openfda
from devicesafe.retrieval import NarrativeIndex, problem_frequencies
from devicesafe.review import build_prompt, cited_ids, review

NARRATIVES = [
    ("Death", "Pump delivered free flow of oxytocin after the cassette door was opened, causing over-infusion.", ["Free or Unrestricted Flow"]),
    ("Injury", "Occlusion alarm failed to sound; patient received no medication for two hours.", ["Alarm, Audible", "Occlusion Within Device"]),
    ("Malfunction", "Battery depleted quickly during transport and the pump shut down without warning.", ["Battery Problem"]),
    ("Injury", "Nurse programmed 100 mL/h instead of 10 mL/h; dose error reduction software was not enabled.", ["Use of Device Problem"]),
    ("Malfunction", "Occlusion alarm did not activate when the line was clamped during testing.", ["Alarm, Audible"]),
]


def raw(i, etype, text, problems):
    return {"report_number": f"3001234-2026-{i:05d}", "date_received": f"2026010{i + 1}",
            "event_type": etype, "product_problems": problems,
            "device": [{"brand_name": "PumpX", "generic_name": "INFUSION PUMP",
                        "manufacturer_d_name": "ACME"}],
            "mdr_text": [{"text_type_code": "Description of Event or Problem", "text": text},
                         {"text_type_code": "Additional Manufacturer Narrative", "text": "Evaluation pending."}]}


FAKE = [raw(i, *n) for i, n in enumerate(NARRATIVES)]


@pytest.fixture
def events(monkeypatch):
    def fake_get(params, session=None):
        if "count" in params:
            return {"results": [{"term": "Malfunction", "count": 2}]}
        skip, limit = int(params["skip"]), int(params["limit"])
        return {"results": FAKE[skip:skip + limit]}
    monkeypatch.setattr(openfda, "_get", fake_get)
    return openfda.fetch_events("infusion pump", limit=10)


def test_build_search_exclude():
    q = openfda.build_search("infusion pump", exclude=["insulin"])
    assert q.endswith('+AND+NOT+device.generic_name:"insulin"')


def test_build_search():
    q = openfda.build_search('infusion "pump"', since="20250101")
    assert q.startswith('device.generic_name:"infusion pump"') and "20250101+TO+" in q


def test_parse_prefers_event_description(events):
    assert len(events) == 5
    assert events[0].narrative.startswith("Pump delivered") and "Evaluation pending" not in events[0].narrative


def test_pagination(monkeypatch, events):
    monkeypatch.setattr(openfda, "MAX_PER_PAGE", 2)
    assert len(openfda.fetch_events("infusion pump", limit=4)) == 4


def test_retrieval_finds_alarm_reports(events):
    hits = NarrativeIndex(events).search("occlusion alarm failure", k=2)
    assert all("alarm" in e.narrative.lower() for e, _ in hits)
    assert problem_frequencies(events)[0] == ("Alarm, Audible", 2)


def test_citation_check(events):
    hits = NarrativeIndex(events).search("alarm", k=3)
    real = hits[0][0].report_number
    out = cited_ids(f"Alarms fail [{real}] and [9999999-2020-00001].", hits)
    assert out == {"valid": [real], "unknown": ["9999999-2020-00001"]}


def test_review_with_fake_llm(events):
    hits = NarrativeIndex(events).search("alarm", k=3)
    rn = hits[0][0].report_number
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"Alarm failures recur [{rn}].")])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    out = review("infusion pump", events, "alarm failures", client=client)
    assert out["mode"] == "llm" and out["citations"]["valid"] == [rn]
    assert "Use ONLY the reports provided" in seen["system"]
    assert rn in seen["messages"][0]["content"]


def test_review_extractive_without_key(events, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = review("infusion pump", events)
    assert out["mode"] == "extractive" and out["sources"]


def test_prompt_contains_context(events):
    hits = NarrativeIndex(events).search("battery", k=2)
    p = build_prompt("infusion pump", "battery?", events, hits)
    assert "Death (1)" in p and "Battery Problem" in p


def test_api(events, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import api.main as api
    api._events.cache_clear()
    c = TestClient(api.app)
    r = c.get("/events", params={"device": "infusion pump", "limit": 10})
    assert r.status_code == 200 and r.json()["fetched"] == 5
    r = c.post("/review", json={"device": "infusion pump", "question": "alarm failures", "limit": 10})
    assert r.status_code == 200 and r.json()["mode"] == "extractive"
    assert c.get("/events", params={"device": "x", "since": "2025"}).status_code == 422
