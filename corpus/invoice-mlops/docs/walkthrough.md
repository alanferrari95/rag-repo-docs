# Walkthrough

A tour of every component: what it is, where it lives, why it is built this way, and what an interviewer is likely to ask (with a short answer). Read it alongside the code; each section names the files.

Contents: [1 Data](#1-data-and-labels) · [2 Training](#2-training) · [3 Features in BigQuery](#3-features-in-bigquery) · [4 Pipeline and gate](#4-pipeline-and-promotion-gate) · [5 Model Registry](#5-model-registry) · [6 Serving](#6-serving-api) · [7 Validation and logging](#7-input-validation-and-prediction-logging) · [8 CI/CD](#8-cicd-with-cloud-build) · [9 Releases](#9-canary-and-rollback) · [10 Continuous training](#10-continuous-training-scheduler-and-launcher) · [11 Drift](#11-drift-monitoring) · [12 Extraction](#12-engine-b-extraction) · [13 Extraction evaluation](#13-extraction-evaluation-and-prompt-gate) · [14 Terraform and IAM](#14-terraform-and-least-privilege) · [15 SHAP](#15-explainability-shap) · [16 Tests](#16-tests-without-the-cloud) · [Pitch](#the-pitch) · [Gold-set review checklist](#gold-set-review-checklist)

---

## 1. Data and labels

**What.** 400 synthetic invoices (`src/features/generate_dataset.py`) and a 120-invoice gold set (`data/gold/risk_gold.csv`, seed 7, no label noise).

**How.** Vendors get a typical amount; each invoice draws around it (lognormal). 12% get a mismatch between total and lines, 12% arrive at night. The label is a rule (ratio > 2.5, mismatch, new vendor, night) and then 8% of labels are flipped.

**Why.** No client data, ever ([PII note](pii.md)). Bootstrapping labels from rules and letting a model generalise is a real technique for cold-starting a review queue. The flips make the problem non-trivial and stand in for human labelling error. The gold set has no flips, so it measures the policy, not the noise.

**Interview questions.**
- *"Your labels come from rules. Why not just ship the rules?"* Rules are the cold start. The model learns interactions (e.g. a high ratio matters less for a vendor with volatile amounts) and gets retrained as reviewed outcomes come in. The rules become one input, not the product.
- *"Why is holdout F1 (0.67) so far below gold F1 (0.86)?"* The holdout labels carry the 8% noise, so even a perfect learner of the policy is penalised there. The gold set has clean labels. That gap is the reason to keep a curated gold set apart from the training distribution.
- *"How would you get real labels?"* Log the analyst's decision for each reviewed invoice (the `predictions` table already keys on time and model version), then join it back for evaluation and retraining.

## 2. Training

**What.** `src/train/train.py`: `fit()` (stratified 80/20 split, XGBoost, holdout metrics) and `save()` (`model.joblib` + `metrics.json`). The same code runs as a CLI (laptop, Vertex Training custom job) and inside the pipeline.

**Why.** XGBoost is the standard for small tabular data, trains in seconds on CPU, and TreeSHAP gives exact explanations. Versions are pinned (`requirements-*.txt`) because the API unpickles what the training image wrote; a scikit-learn or XGBoost version mismatch between images is a classic silent bug.

**Vertex AI Training and Experiments** (`scripts/run_experiments.py`): each hyperparameter config runs as a CPU custom job on the training image, then its params, holdout metrics and gold metrics go into one Vertex Experiments run (`risk-hparams`) for side-by-side comparison.

**Interview questions.**
- *"Why a custom container and not a prebuilt training container?"* Same image for custom jobs and pipeline steps, so what you tested is what runs, with our pinned versions.
- *"How do you make training reproducible?"* Fixed seeds, pinned dependencies, data versioned in git (and in BigQuery with the load script), the image tag recorded with the run, metrics written next to the model.

## 3. Features in BigQuery

**What.** `src/features/features.sql` computes `vendor_avg_amount` (the vendor's mean over its other invoices) and the ratio with window functions over the raw `invoices` table. `build_features.compute_features()` is the pandas twin, and a test checks they produce the committed dataset.

**Why.** The data lives in BigQuery, so computing features there avoids moving it and keeps one definition. A managed Feature Store is overkill at this size.

**Interview questions.**
- *"What is training-serving skew here?"* Offline, the average is leave-one-out over *all* the vendor's invoices, future ones included. Online, the caller sends `vendor_avg_amount`. If they compute it differently (only past invoices, another vendor set), the model sees a different distribution. The fix is a point-in-time feature (`ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`) used both offline and online. The PSI check would catch the symptom.
- *"How do you keep SQL and Python in sync?"* The parity test. In a bigger system, generate both from one definition, or only compute features in one place and serve them.

## 4. Pipeline and promotion gate

**What.** `pipelines/risk_train_pipeline.py` (KFP v2 on Vertex AI Pipelines): `features-bq` → `train-xgboost` → `evaluate-gold` → `dsl.If` promote → `register-model`, `dsl.Else` → `notify-rejected`. An `ExitHandler` publishes any task failure to Pub/Sub. Gate logic is in `src/eval/gate.py`.

**How the gate decides.** The candidate's gold F1 must be ≥ `min_f1` (0.75) and ≥ champion F1 − `max_regression` (0.02). The champion is the version holding the alias, re-scored on the same gold set in the same run, so the comparison is apples to apples even if the gold set changed.

**Why.** Absolute thresholds alone let quality decay slowly. Champion comparison alone lets a bad first model in. Together they mean "good enough, and not worse than what we have". Rejection is a normal outcome (`Else` branch), not a pipeline failure, so alerts separate "the model was not good enough" from "the pipeline broke".

**Interview questions.**
- *"Why is caching disabled on the gate?"* With caching, a re-run with the same inputs would reuse the old decision even though the champion changed. The gate and the registration must look at the Registry now.
- *"What is lineage and where do you see it?"* Features, gold set and model are KFP artifacts, so Vertex ML Metadata links each Registry version to its run, its data and its metrics. Click the model version in the console to see the pipeline that produced it.
- *"Why are the components thin?"* Logic in `src/` is unit-tested without KFP; components only wire inputs and outputs. One image for all components avoids "works in step 2, breaks in step 3" dependency drift.

## 5. Model Registry

**What.** `src/registry.py`. Two Registry entries with fixed ids: `risk-classifier` (model artifacts) and `invoice-extractor` (prompt + evaluation report). Versions are uploaded by the pipelines, and the `production` / `staging` aliases point at the version in use.

**How serving uses it.** At deploy time Cloud Build resolves `production` to a version id (`scripts/registry_version.sh`) and pins it on the revision (`MODEL_VERSION_ID`). At startup the API fetches that version's `artifactUri` through the REST API (`google-auth`, without the heavy SDK) and loads `model.joblib`.

**Why pin per revision instead of reading the alias at startup?** With `min-instances=0`, instances start at different times. If each one read the alias at startup, one revision could serve two models at once. Pinning makes a revision immutable: rolling back traffic also rolls back the model.

**Interview questions.**
- *"How does a newly promoted model reach production without a code change?"* The register step publishes `model-promoted`. A Pub/Sub-triggered Cloud Build (`cloudbuild-model-deploy.yaml`) creates a revision with the same image pinned to the new version, at 20%. Same canary and rollback as code.
- *"Why aliases and not 'latest'?"* "Latest" is whatever was uploaded last, including staging candidates. An alias is an explicit, auditable promotion that can be moved back.

## 6. Serving API

**What.** `src/api/main.py`, built by `create_app(model_loader, sink, extractor_factory)`. Endpoints: `/score`, `/extract`, `/version` (app, model, prompt, revision), `/health`.

**Why a factory.** Every dependency can be injected, so tests run the real app with an in-memory model, sink and LLM. In production the defaults read env vars: Registry, BigQuery, Gemini.

**Why Cloud Run and not a Vertex Endpoint.** Scale to zero, native traffic splitting and tag URLs, pay per use. An Endpoint bills a node every hour even when idle. Worth it for GPUs, very high QPS or Vertex's built-in monitoring and explanations.

**Interview questions.**
- *"What happens on a cold start?"* The instance downloads the model from GCS and loads it. In the September load test, the only 5xx (6 of 200) were cold starts. `min-instances=1` fixes that and bills an idle instance.
- *"Why is logging synchronous?"* Cloud Run may throttle CPU after the response is sent, so background tasks are not reliable without "CPU always allocated". The cost is a few ms per call; the latency field measures the model only.

## 7. Input validation and prediction logging

**What.** Pydantic checks types, Pandera checks ranges and allowed values (`ScoreSchema`): a bad row gets a 422 before it reaches the model. Each valid prediction writes one row to `predictions`: features, score, model version, model latency.

**Why both.** Pydantic answers "is this JSON shaped right?", Pandera answers "is this data plausible?" (amount > 0, hour in 0–23, binary flags). Silent acceptance of corrupt input is worse than an error.

**Interview questions.**
- *"Logging failed, what does the client see?"* A normal answer. The failure goes to Cloud Logging (`test_logging_failure_does_not_break_scoring`). You lose one row of monitoring, not a business request.
- *"Why log the model version on every row?"* It lets you compare canary vs. stable in SQL, and lets the drift job compare only rows from the model it is judging.

## 8. CI/CD with Cloud Build

**What.** `cloudbuild-pr.yaml`: on PR, install pinned deps, run all tests, compile both pipelines. `cloudbuild.yaml`: on merge, the same, then build/push `train`, `api` and `monitor`; deploy a 0% tagged revision; 20% canary; publish gold sets and compiled templates to GCS; update the Cloud Run Jobs' image; submit a staging pipeline run.

**Why compile the pipeline in CI.** A broken pipeline definition fails the PR, not the Monday run (week 9's deliberate schema-bug PR failed red). Publishing the compiled template on merge means the scheduled run uses exactly the merged code.

**Interview questions.**
- *"CI vs CD vs CT?"* CI tests and builds code; CD ships it (here as a canary); CT retrains the model on new data on a trigger. They have different triggers and gates.
- *"Why does the PR build have its own service account?"* PR builds run code from any branch. The `ci` SA can only write logs, so a malicious or buggy PR cannot deploy or read data.

## 9. Canary and rollback

**What.** `scripts/canary.sh`, `promote.sh`, `rollback.sh`, all `gcloud run services update-traffic`. Runbook in `docs/runbook.md`.

**Bug worth telling.** The first `rollback.sh` pinned traffic to a hard-coded revision. While traffic is pinned, `gcloud run deploy` creates new revisions at 0%, so two merges shipped nothing for weeks while the build stayed green. The fix: deploys are explicitly `--no-traffic` + canary, and rollback reads the stable revision from the live split.

**Interview questions.**
- *"Why is rollback instant?"* Revisions are immutable and already deployed; rollback only moves traffic. No rebuild, no redeploy.
- *"How do you decide to promote?"* Compare canary vs. stable: error rate and latency per revision (Cloud Logging), score distribution per model version (BigQuery). With more traffic, automate it with thresholds.

## 10. Continuous training: Scheduler and launcher

**What.** `src/launcher.py` runs as the `risk-pipeline-launcher` Cloud Run Job. Cloud Scheduler starts it every Monday at 08:00 (`infra/monitoring.tf`). It submits the compiled template from GCS with alias `production`.

**Guards.** Never submit while a `risk-train-eval` run is queued or running. After a drift-triggered run, wait 24 h before the next drift-triggered one.

**Why a job and not Scheduler calling Vertex directly.** Scheduler calling the Vertex REST API would have to carry the whole compiled spec in the job body (or use a template in Artifact Registry), and the guards need code anyway. A Cloud Run Job keeps it in one tested Python function shared by the weekly run and the drift job.

**Interview questions.**
- *"What stops a retraining storm?"* The two guards, plus the gate: retraining on bad data still has to beat the champion on the gold set.
- *"Why not retrain on every drift alert?"* Drift in inputs is not always harm; it can be a seasonal shift the model handles. Here the drift job retrains because retraining is cheap and gated. With expensive training, alert a human first.

## 11. Drift monitoring

**What.** `src/monitoring/drift.py` as the `drift-check` Cloud Run Job, daily at 06:00. It compares the last 24 h of predictions from the production model version against the training data: **PSI** per feature and a two-sample **KS test** on the score. On drift it calls the launcher, then publishes a report to `invoice-mlops-drift-alerts`.

**How PSI works.** Cut the reference into 10 quantile bins, compute the share of reference and production rows per bin, then sum `(p − r) · ln(p / r)`. Below 0.1 is stable, 0.1–0.25 is a moderate shift, above 0.25 is a major shift (threshold used). Duplicate bin edges collapse, so binary features get 2 bins instead of crashing.

**Why KS on the score too.** Features can each shift a little while the combination moves the output a lot. The score is what the business consumes.

**Validation.** `scripts/validate_drift.sh`: normal traffic sampled from the training data, then skewed traffic (amounts ×8, night hours, only new vendors). Only the second batch should alert and start a run. `tests/test_drift_job.py` does the same end to end with fakes.

**Interview questions.**
- *"Data drift vs. concept drift?"* Data drift: P(X) changes; you can see it without labels (this job). Concept drift: P(y|X) changes; you need labels (analyst outcomes) to see it.
- *"Why filter by model version?"* After a promotion, yesterday's scores came from another model. Comparing them with today's reference scores would flag drift that is only a model change.
- *"What if there is little traffic?"* Below 50 rows the job reports `insufficient_data` instead of a verdict. PSI on 10 rows is noise.

## 12. Engine B: extraction

**What.** `src/extract/`: `pages.py` (page filter), `extractor.py` (provider-agnostic), `gemini.py` and `ollama.py` (backends), `schema.py` (`InvoiceFields`), `prompts/v1.txt`, `v2.txt`. Exposed as `POST /extract`.

**Page filter.** Score every page's text with weighted keyword patterns (invoice/factura, totals, VAT/IVA, invoice-number labels) and negative ones (terms and conditions, remittance advice, "Dear customer"). Keep pages above the threshold, and keep all pages if none qualifies. On the gold set it keeps exactly the invoice page in 40/40 documents, so 40 of 117 pages and 47% of the text reach the LLM.

**Structured output.** The LLM must answer the `InvoiceFields` JSON schema (Gemini `response_schema`, Ollama `format`), with temperature 0. Pydantic validates the answer either way.

**Backends.** `EXTRACT_BACKEND=vertex` (Gemini on Vertex AI, Google credentials, no key), `gemini-api` (API key), `ollama` (local model), `off`. The default model is `gemini-3.8-flash` on the `eu` endpoint, because no GA Flash model was served from `europe-west4` regionally on 2026-10-22.

**Logging.** One row per call in `extractions`: document hash, pages, tokens (prompt, output, thinking), latency, model, prompt version, number of fields found. Never the document or the values.

**Interview questions.**
- *"Why filter pages instead of sending the whole PDF?"* Input tokens are the cost and latency driver, and distractor pages (previous balances, PO numbers, IBANs) are where wrong answers come from. It is also data minimisation.
- *"Text or images?"* These PDFs have a text layer, so text is cheaper. Scans need OCR or multimodal input. The filter would then run on OCR text.
- *"How do you stop hallucinated values?"* The schema allows null, the prompt says "never guess", and evaluation counts a value where the gold has null as a false positive.

## 13. Extraction evaluation and prompt gate

**What.** `src/eval/extract_metrics.py` (per-field precision/recall), `src/eval/evaluate_extract.py` (runs a backend over a labelled set; reports quality, tokens, latency p50/p95 and cost per 1,000 documents), `pipelines/extract_eval_pipeline.py` (gate + Registry).

**Definitions.** After normalisation (case and punctuation for names; ±0.01 for amounts; exact ISO date and currency code):
- TP: predicted and right; FP: predicted and wrong (including a value where gold is null); FN: gold has a value and the prediction is missing or wrong.
- Precision = TP/(TP+FP): when it answers, is it right? Recall = TP/(TP+FN): how much of what is there does it get?
- A wrong value counts as both an FP and an FN.

**Gate.** Micro precision ≥ 0.85, micro recall ≥ 0.80, micro F1 not more than 0.02 below the production prompt. `v1` (one line) is there to be rejected: it gives no rule for date format, currency codes or which total to take. `v2` adds those rules.

**Interview questions.**
- *"Why per field and not one accuracy number?"* Fields fail differently: currency is easy, the vendor in a footer is hard. A single number hides which field regressed.
- *"Why is precision the hard gate?"* A wrong total that flows into payment is worse than a blank that goes to a human. Recall has a lower bar.
- *"MLOps vs LLMOps?"* Here: engine A retrains on data and monitors input drift; engine B changes when the prompt or the provider model changes, so it is re-evaluated on every change, versioned the same way, but has no training loop.

## 14. Terraform and least privilege

**What.** `infra/`: APIs, buckets, Artifact Registry, BigQuery, Cloud Run, Pub/Sub, Cloud Run Jobs, Cloud Scheduler, Cloud Build triggers, IAM. State in a versioned GCS bucket. Works in any project via `terraform.tfvars`. Resources that predate Terraform in the original project are imported (`infra/examples/`), not recreated.

**Ownership split.** Terraform owns existence, identity, scaling and IAM. Cloud Build owns what changes on every release (image, pinned versions, traffic). The Cloud Run resource ignores those fields so the two never fight.

**Identities.** `work` (runtime), `build` (deploy, `run.developer`, no IAM rights), `ci` (logs only), `scheduler` (start two jobs). No JSON keys anywhere: workloads get short-lived tokens from the metadata server, humans use `gcloud auth login`. The Compute Engine default SA's project `Editor` is removed.

**Interview questions.**
- *"Why not one SA for everything?"* Blast radius. A leaked PR build should not deploy; the scheduler should not read data.
- *"Why no `--allow-unauthenticated` in the deploy?"* It calls `setIamPolicy`, so the build SA would need `run.admin`. Public access is set once in Terraform (`invoker_iam_disabled`), and the build keeps `run.developer`.
- *"How do you change something created by hand?"* `import` blocks, then `plan` must show no changes before you add anything.

## 15. Explainability (SHAP)

**What.** `scripts/shap_report.py` writes three plots (`docs/img/`): global importance, beeswarm, and a waterfall for one high-risk invoice. `POST /score?explain=true` returns per-feature SHAP values from XGBoost itself (`pred_contribs`, exact TreeSHAP) plus the base value; base + sum = the model's log-odds (tested).

**Reading the plots.** `is_new_vendor` and `hour` dominate, which matches the labelling rules: the model learned the policy it was given. If a feature the rules never mention ranked high, that would point to leakage or a data bug.

**Interview questions.**
- *"SHAP vs. feature importance?"* Gain importance is global and biased toward high-cardinality splits. SHAP is per prediction, additive and consistent; averaging |SHAP| gives a global view.
- *"Why compute it in the API without the shap library?"* XGBoost already implements exact TreeSHAP; no extra dependency in the serving image.

## 16. Tests without the cloud

73 tests run in CI and on a laptop with no GCP access: fakes for BigQuery (sink), Pub/Sub and Vertex (monkeypatched functions), the Registry (fake fetch), and the LLM (a fake Gemini client that answers from the gold labels only if the page filter kept the right page; a fake Ollama transport). The pipelines are compiled and their structure is asserted. The Cloud Build YAMLs are parsed. The PR build also runs `terraform fmt -check` and `terraform validate` offline (no backend, no credentials).

**Interview question.** *"What do your tests not cover?"* IAM and networking (only a real `plan`/`apply` shows those), the actual LLM quality (only an evaluation run against the provider shows it), and Vertex runtime behaviour (artifact paths, FUSE mounts). Those need the cloud runs listed in `docs/evidence.md`.

---

## The pitch

> "In consulting I industrialised document intelligence for enterprise clients. To own the whole lifecycle, I built invoice-mlops: two engines on GCP. A risk classifier retrains itself: features in BigQuery, a Vertex AI pipeline that only promotes a model if it beats the current one on a gold set, Model Registry, canary releases on Cloud Run with instant rollback, and a daily drift job that retrains when the inputs move. Next to it, a Gemini extraction module with page filtering and structured output, scored field by field, where a pipeline gate rejects worse prompt versions. Everything is in Terraform with least-privilege service accounts and no keys."

## Gold-set review checklist

The 40 extraction labels are written by the generator, so they are correct for what it meant to print. A human pass checks what a reader actually sees. For each `data/gold/extract/docs/gold-NNN.pdf`, open the PDF next to its line in `labels.jsonl`:

1. `vendor_name`: is it the issuer (not the customer), and is it visible as written? In "tax" layouts it is only in the small footer.
2. `invoice_number`: matches the number after "Invoice No" / "Nº de factura" / "Reference", or is `null` when there is none. Not the PO number on the cover letter.
3. `invoice_date`: the issue date (not the due date) in ISO format.
4. `total_amount`: the final total including VAT, not the subtotal or the "previous balance".
5. `currency`: ISO code matching the symbol or code on the page.
6. `invoice_page`: the page that holds the invoice (used by the page-filter test).

If every line passes, the gold set is "hand-verified". Record the date and reviewer in `data/gold/extract/README.md`.
