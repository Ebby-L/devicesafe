# 📋 DeviceSafe

**Grounded, citation-checked design-risk reviews from FDA medical device adverse event reports.**

The FDA's MAUDE database holds a constant stream of reports about devices that malfunctioned or were involved in patient harm. For a device designer, that is free evidence about what goes wrong in real use, but it sits in unstructured narrative text. DeviceSafe turns a design question into a short risk review, framed in ISO 14971 terms (hazard → hazardous situation → harm), with every claim tied to the report that supports it.

**Case study:** general-purpose IV infusion pumps (insulin pumps excluded), which deliver oxytocin for the prevention and treatment of postpartum hemorrhage.

## How it works

```
design question ──► openFDA API ──► report narratives ──► TF-IDF retrieval (top k)
                                                                  │
         citation check ◄── risk review ◄── LLM (Claude) ◄────────┘
     (flags any report number                (cites report numbers,
      that wasn't supplied)                   stays inside the evidence)
```

1. **Fetch:** pulls recent adverse event reports for a device type from openFDA, plus server-side counts across all matching reports.
2. **Retrieve:** ranks narratives by relevance to the question, so the LLM sees only the evidence that matters.
3. **Synthesise:** the LLM writes recurring failure modes, hazard chains, design implications and evidence gaps. It is instructed to cite every claim and never infer incidence rates, which MAUDE cannot support.
4. **Verify:** every cited report number is checked against the supplied reports, so invented references are flagged.

Without an API key, DeviceSafe falls back to an extractive summary, so the whole pipeline still runs for free.

## What's inside

| Part | What it does |
|---|---|
| `notebooks/devicesafe_infusion_pumps.ipynb` | The infusion pump case study end to end |
| `devicesafe/openfda.py` | openFDA client with pagination and server-side counts |
| `devicesafe/retrieval.py` | Narrative index and problem frequencies |
| `devicesafe/review.py` | Prompting, LLM call, extractive fallback, citation check |
| `api/main.py` | FastAPI service: `/events` and `/review` |
| `tests/` | Offline tests with faked openFDA responses and a faked LLM |

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # optional; enables the LLM review

python -m devicesafe "infusion pump" -q "What causes over-infusion?"
uvicorn api.main:app --reload       # API docs at http://127.0.0.1:8000/docs
pytest
```

Or open the notebook in Google Colab and add `ANTHROPIC_API_KEY` under Secrets.

## Example output

<!-- After running the notebook, paste an excerpt of a review here. -->
_An excerpt of a generated review goes here after the notebook is run._

## Limitations

- MAUDE reports are unverified and under-reported. They show what *can* go wrong, never how often.
- Generic-name search misses some relevant reports and includes a few unrelated ones.
- Retrieval is lexical; embedding-based retrieval would catch paraphrased descriptions.
- The LLM synthesises but does not verify. Conclusions still need an engineer's reading of the source reports.

## Next steps

- Embedding retrieval with re-ranking
- Export findings into a structured hazard log for design reviews
- Extend to devices central to low-resource maternity care: fetal monitors, BP monitors, pulse oximeters

## Credits

Data: [openFDA](https://open.fda.gov/), U.S. Food and Drug Administration. openFDA data is unverified and not for clinical decision-making.

---

