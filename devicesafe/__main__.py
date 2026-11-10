"""Command line: python -m devicesafe "infusion pump" --question "..." """
import argparse

from .openfda import fetch_events
from .review import review


def main() -> None:
    ap = argparse.ArgumentParser(description="Design-risk review from FDA adverse event reports")
    ap.add_argument("device", help='Generic device name, e.g. "infusion pump"')
    ap.add_argument("--question", "-q")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--since", help="YYYYMMDD")
    a = ap.parse_args()

    events = fetch_events(a.device, limit=a.limit, since=a.since)
    print(f"Fetched {len(events)} reports for '{a.device}'.\n")
    if not events:
        return
    out = review(a.device, events, a.question)
    print(out["answer"])
    if out["citations"]["unknown"]:
        print("\n⚠ Citations not found in the supplied reports:", ", ".join(out["citations"]["unknown"]))


if __name__ == "__main__":
    main()
