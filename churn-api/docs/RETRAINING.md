# Retraining and redeploying safely

Principles: a model version is **immutable** (never overwritten), every change goes through the same gates a first release did, the old version stays one setting away, and a human signs off on the decision threshold.

## 1. When to retrain
* **Scheduled**: monthly or quarterly. Churn labels arrive late (a customer is "churned" only after leaving), so the freshest usable labels are a few weeks old.
* **Triggered**: `POST /drift` on last week's inputs flags features (PSI > 0.2) two weeks running, or live outcome metrics fall (recall or cost per customer, from the labelled predictions of 2–3 months ago), or the cost inputs change (new offer price, new customer-value estimate).

## 2. Build a candidate (nothing live is touched)
1. Export a fresh snapshot; record its SHA-256. Validate before training: schema, row-count change < 20 %, churn rate within ±5 points of the last run, no duplicate IDs, share of blanks per column. Fail loudly, do not repair silently.
2. `python -m training.train --data <snapshot>`; this tunes, picks the threshold from out-of-fold predictions, and writes `artifacts/<new-version>/` with data hash, model hash, library versions, metrics and drift reference. Same code, same seed, so the run is reproducible.
3. Commit the code and the new artefact directory in one pull request so the review sees the metrics next to the diff.

## 3. Gates the candidate must pass (automated in CI, on a holdout newer than the training data)
| Gate | Rule |
|---|---|
| Quality vs. the live model | ROC-AUC not lower by more than 0.01 (paired bootstrap on the **same** rows for both models) |
| Business metric | expected cost per customer no worse than live model, using the **current** cost inputs |
| Calibration | Brier score within 10 % of live; calibration curve within 0.05 of the diagonal in each decile |
| Segments | recall for month-to-month and for tenure ≤ 12 months not worse than live by > 0.03 |
| Contract | `pytest` (82 tests incl. API contract and the 100 ms median latency check), OpenAPI file unchanged or intentionally regenerated |
| Image | builds; size < 600 MB; container answers `/health`, one `/predict`, and rejects a request with no key |

A failing gate blocks the merge. A passing gate is not an approval: the product owner also reviews the **threshold** and the cost assumptions behind it.

## 4. Roll out in stages
1. **Shadow.** Deploy the candidate beside the live model. The CRM still uses the live answers; both scores are logged under their `model_version`. Compare score distributions and agreement for a few days.
2. **Canary.** Route 5 % of CRM calls (or a fixed customer subset) to the candidate, then 25 %, then 100%, pausing between steps. Advance only while error rate, p95 latency and the share of customers flagged stay inside agreed bounds. For a true effect measurement, hold out a random control group that receives no offer.
3. **Promote.** Change `artifacts/CURRENT` (or `CHURN_MODEL_DIR`) to the new version and redeploy. The previous version stays in the image.

## 5. Rollback
* Set `CHURN_MODEL_DIR=/app/artifacts/<previous-version>` and restart: seconds, no rebuild. Or redeploy the previous image tag.
* Every response and every log line carries `model_version`, so you can tell which decisions the bad model made and re-score them.
* Rollback triggers: 5xx rate or p95 latency above the bound, flagged share moving more than ±10 points overnight, a PSI alert on model **outputs**, or a data-quality incident upstream.

## 6. After release
* Keep the sealed-test discipline: the holdout used to approve this version must not be used to tune the next.
* Log (customer id, `model_version`, score, decision, offer made, outcome) so next quarter's retraining can estimate the real offer success rate `s` from the control group and the threshold can be re-derived from measurement, not assumption.
* Rotate API keys on a schedule; keys are configured as `sha256:<digest>` entries so plaintext never sits in configuration.
* Dependency upgrades are retraining events: the artefact is bound to the exact scikit-learn version it was trained with (the service logs a warning when they differ).
