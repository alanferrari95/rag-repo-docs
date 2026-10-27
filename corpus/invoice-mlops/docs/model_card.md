# Model card: invoice review risk (`risk-classifier`)

**What it predicts.** A binary label: should this invoice go to an analyst before it is paid (1) or not (0)? A 1 means "inconsistent or unusual enough to justify a human look". **It does not predict fraud.**

**Model.** XGBoost (`n_estimators=100`, `max_depth=4`) on 6 features: `amount`, `lines_sum`, `hour`, `is_new_vendor`, `vendor_avg_amount`, `amount_vs_avg_ratio`. Identifiers (`invoice_id`, `vendor_id`, `vendor_name`) are never features.

**Data.** 400 synthetic invoices from 40 vendors (`src/features/generate_dataset.py`, seed 42). Labels are bootstrapped from rules (amount far above the vendor's average, total ≠ sum of lines, new vendor, night-time submission), then **8% are flipped** to simulate analyst mistakes. Features are computed in BigQuery (`src/features/features.sql`); `vendor_avg_amount` is the vendor's mean over its *other* invoices.

**Evaluation.**

| Set | Size | Labels | F1 | Precision | Recall |
|---|---|---|---|---|---|
| holdout (20% stratified split) | 80 | noisy, like training | 0.667 | 0.773 | 0.586 |
| gold set `data/gold/risk_gold.csv` | 120 | clean (no flips), different seed | 0.863 | 0.957 | 0.786 |

Local run, `python scripts/local_report.py`. The gold set measures how well the model learned the intended policy; the holdout is capped by its own label noise. Promotion requires gold F1 ≥ 0.75 and no more than 0.02 below the current production version.

**Explanations.** `scripts/shap_report.py` → [global importance](img/shap_bar.png), [beeswarm](img/shap_beeswarm.png), [one high-risk invoice](img/shap_waterfall_high_risk.png). Mean |SHAP| on the gold set ranks `is_new_vendor` first, then `hour`, `amount_vs_avg_ratio`, `vendor_avg_amount`, `amount`, `lines_sum`: consistent with the labelling rules. Per request: `POST /score?explain=true`.

**Known limits.**
- Synthetic data: the metrics say nothing about real invoices.
- The leave-one-out vendor average uses the vendor's *future* invoices too. That matches how the labels were built, but a production system needs a point-in-time average (only invoices before this one), computed the same way online and offline. Otherwise there is training-serving skew.
- The caller sends `vendor_avg_amount`; if they compute it over a different vendor universe, scores drift. That is what the daily PSI check is for.
- Small data: on 120 gold invoices, one wrong prediction moves F1 by about 0.01.
