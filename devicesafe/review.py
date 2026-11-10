"""Grounded design-risk review: retrieve relevant reports, then have an LLM synthesise them.

Uses the Anthropic API when ANTHROPIC_API_KEY is set. Without a key, falls back to an
extractive summary so the project still runs end to end for free.
"""
from __future__ import annotations

import os
import re

from .openfda import Event
from .retrieval import NarrativeIndex, event_type_counts, problem_frequencies

MODEL = os.getenv("DEVICESAFE_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.getenv("DEVICESAFE_MAX_TOKENS", "8000"))
MAX_CHARS_PER_REPORT = 1200

SYSTEM = """You are a medical device risk engineer reviewing FDA MAUDE adverse event reports \
to inform design and risk management under ISO 14971.

Rules:
- Use ONLY the reports provided. Cite every claim with report numbers in square brackets, e.g. [1234567-2024-00012].
- If the reports do not support an answer, say so plainly. Do not use outside knowledge about specific products or manufacturers.
- MAUDE reports are unverified and under-reported. Never state or imply incidence rates, and never conclude that a device or manufacturer is unsafe.
- Be concise and specific. Write for a design engineer."""

TEMPLATE = """Device under review: {device}
Question: {question}

Context across {n_total} fetched reports:
- Event types: {types}
- Most frequent product problems: {problems}

Most relevant reports:
{reports}

Write the review with these sections:
1. **Recurring failure modes**: patterns that appear in more than one report.
2. **Hazard chains**: for the most serious patterns, hazard → hazardous situation → potential harm.
3. **Design and usability implications**: concrete requirements or controls an engineer could act on.
4. **Evidence gaps**: what these reports cannot tell us."""


def format_report(e: Event) -> str:
    text = e.narrative[:MAX_CHARS_PER_REPORT] + ("…" if len(e.narrative) > MAX_CHARS_PER_REPORT else "")
    problems = ", ".join(e.product_problems) or "none listed"
    return (f"[{e.report_number}] {e.date_received} | {e.event_type} | "
            f"{e.brand_name or 'unbranded'} | problems: {problems}\n{text}")


def build_prompt(device: str, question: str, events: list[Event],
                 hits: list[tuple[Event, float]]) -> str:
    return TEMPLATE.format(
        device=device, question=question, n_total=len(events),
        types=", ".join(f"{k} ({v})" for k, v in event_type_counts(events).items()),
        problems=", ".join(f"{p} ({c})" for p, c in problem_frequencies(events, 8)) or "none listed",
        reports="\n\n".join(format_report(e) for e, _ in hits),
    )


def cited_ids(text: str, hits: list[tuple[Event, float]]) -> dict[str, list[str]]:
    """Split citations in the answer into ones we supplied and ones we did not (hallucinated)."""
    supplied = {e.report_number for e, _ in hits}
    found = set(re.findall(r"\[([0-9A-Za-z\-]{6,})\]", text))
    return {"valid": sorted(found & supplied), "unknown": sorted(found - supplied)}


def _extractive(device: str, events: list[Event], hits: list[tuple[Event, float]]) -> str:
    lines = [f"**Extractive summary for {device}** (no LLM key set)\n",
             "Most frequent product problems:"]
    lines += [f"- {p}: {c}" for p, c in problem_frequencies(events, 5)]
    lines.append("\nMost relevant reports:")
    lines += [f"- [{e.report_number}] {e.event_type}: {e.narrative[:220]}…" for e, _ in hits[:5]]
    return "\n".join(lines)


def review(device: str, events: list[Event], question: str | None = None,
           k: int = 12, client=None) -> dict:
    question = question or f"What are the main design-related failure modes for {device}?"
    hits = NarrativeIndex(events).search(question, k=k)
    if not hits:
        return {"answer": "No reports matched this question.", "sources": [], "mode": "none",
                "citations": {"valid": [], "unknown": []}}

    if client is None and not os.getenv("ANTHROPIC_API_KEY"):
        answer, mode = _extractive(device, events, hits), "extractive"
    else:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(device, question, events, hits)}])
        answer, mode = "".join(b.text for b in msg.content if b.type == "text"), "llm"
        if not answer.strip():
            # Newer models may spend the token budget reasoning before writing; surface that clearly
            raise RuntimeError(
                f"The model returned no text (stop_reason={getattr(msg, 'stop_reason', None)}, "
                f"blocks={[b.type for b in msg.content]}). Try raising DEVICESAFE_MAX_TOKENS.")

    return {
        "answer": answer, "mode": mode,
        "sources": [{"report_number": e.report_number, "date": e.date_received,
                     "event_type": e.event_type, "score": round(s, 3)} for e, s in hits],
        "citations": cited_ids(answer, hits),
    }