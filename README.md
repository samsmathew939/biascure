# BiasCure ⚖️

**An AI fairness auditor — does a hiring AI score the same résumé differently based only on the applicant's name?**

🔗 **Live demo:** https://biascure.streamlit.app/

## What It Does

Paste a résumé **without a name**. BiasCure attaches first names associated with different demographic groups, runs each identical résumé through the **same** hiring AI, and measures the gap in scores. Then it re-runs the résumé **name-blinded** to show the gap shrink — detection *and* a working mitigation, side by side.

## Why I Built It

Hiring AIs are everywhere, and a model can penalize an identical candidate for nothing but their name. BiasCure makes that invisible bias **visible and measurable** — and shows that blinding names is a simple, real fix.

## How It Works

1. You paste a résumé (no name).
2. The app attaches names from two groups — associated with White and Black applicants in the Bertrand & Mullainathan (2004) field experiment *"Are Emily and Greg More Employable than Lakisha and Jamal?"*
3. It scores every identical résumé through the same AI (0–100 interview-worthiness).
4. It reports a **Bias Scorecard**: mean-score gap + interview-rate disparity (demographic parity).
5. It re-runs the résumé **name-blinded** and shows how much the gap closes.

## Tech Stack

- **Frontend:** Streamlit (Python web app)
- **Scorer (the audited AI):** Llama-3 via the free Groq API (OpenAI-compatible) — provider-agnostic; swap the endpoint to audit any model
- **Fairness metrics:** mean-score gap + selection-rate disparity (demographic-parity difference), computed directly
- **Hosting:** Streamlit Community Cloud (free)

## Honest Limits

- It audits **the one model it's pointed at** — not "all AI."
- Small samples are noisy; results are **illustrative**, not a legal audit.
- It doesn't "cure" bias. It **detects** it and shows that **one mitigation (name-blinding) reduces it**. The name is a hook.
- Names reflect *perceived* demographic associations measured by the 2004 study — not facts about any individual.

## Status

✅ **Live and working** at https://biascure.streamlit.app/ — real Groq-hosted scorer, end-to-end: paste a résumé → scorecard → name-blind mitigation. Runs in **DEMO MODE** (simulated scores) if no key is set; the live app has a key connected. The app auto-selects whichever model the key can access, so it survives Groq retiring model ids.

**Roadmap:**
- [x] Pairing engine (name-swapped identical résumés)
- [x] Bias Scorecard (mean-score gap + interview-rate disparity)
- [x] Name-blinding mitigation with before/after gap
- [x] Public deployment on Streamlit Cloud
- [x] Live Groq scorer connected (free API key)

## Quick Start

```bash
git clone https://github.com/samsmathew939/biascure.git
cd biascure
pip install -r requirements.txt
streamlit run app.py          # runs in DEMO MODE without a key
```

To run a **real** audit, get a free key at console.groq.com (no card), then either:
- set `GROQ_API_KEY` as an environment variable, or
- add it to `.streamlit/secrets.toml` as `GROQ_API_KEY = "your-key"` (local) or in Streamlit Cloud → app settings → Secrets (deployed).

## What's Next

- More demographic axes (gender-coded names, age proxies)
- Audit multiple models side by side
- Export the scorecard as a shareable report

---

Built by Sam Mathew — open source.
