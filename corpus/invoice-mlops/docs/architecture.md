# Architecture

One repository, one GCP project, one region (`europe-west4`), two engines that share the API, the logging and the release process.

| | Engine A: risk model | Engine B: field extraction |
|---|---|---|
| Discipline | classic MLOps: a model we train | LLMOps: a model we prompt |
| What changes over time | the data, so we retrain | the prompt and the provider model, so we re-evaluate |
| Quality gate | F1 on a gold set, no regression vs. the champion | per-field precision/recall on a gold set, no regression vs. the champion |
| Versioned in | Model Registry `risk-classifier` (model artifact) | Model Registry `invoice-extractor` (prompt + evaluation report) |
| Drift | yes: PSI/KS on live features and scores | not monitored here (no labels in production); the gate runs on every prompt change |

## 1. Training loop (continuous training)

```
BigQuery invoices ──features.sql──▶ features (Dataset artifact)
                                        │
                                        ▼
                                 train-xgboost ──▶ model (Model artifact) + holdout metrics
                                        │
gold set (GCS, imported artifact) ──▶ evaluate-gold ──▶ decision
                                        │
                   ┌────────────────────┴───────────────────┐
              promote                                     reject
                   │                                        │
     register-model: new Registry version,          notify-rejected: Pub/Sub
     move alias (staging | production)
                   │
     production? ──▶ Pub/Sub model-promoted ──▶ Cloud Build model-deploy ──▶ 20% canary
```

- **Triggers.** A merge to `main` submits a staging run (alias `staging`, so a code change never promotes a model by itself). Cloud Scheduler starts the `risk-pipeline-launcher` Cloud Run Job every Monday at 08:00 (alias `production`). The drift job calls the same launcher when it finds drift. Any task failure publishes to `invoice-mlops-pipeline-alerts` from the exit handler.
- **Gate.** The candidate is scored on a gold set it never trained on (clean labels, different seed). It must reach `min_f1` (0.75) and must not lose more than `max_regression` (0.02 F1) against the version that currently holds the alias, scored on the same gold set in the same run. Caching is disabled for the gate and the registration, so they always see the Registry as it is now.
- **Lineage.** Features, gold set, model and registered version are KFP artifacts, so Vertex ML Metadata links each Registry version back to its data and its run.
- **One image.** Every component runs on `Dockerfile.train`. Components are thin wrappers around functions in `src/`, which unit tests cover without KFP.

## 2. Serving

```
client ──▶ Cloud Run invoice-mlops-api (service account: invoice-mlops-work)
              ├─ /score   Pydantic → Pandera → XGBoost (+ TreeSHAP with ?explain=true) ──▶ BigQuery predictions
              ├─ /extract PDF → page filter → LLM (structured output) ──▶ BigQuery extractions (metadata only)
              ├─ /version app version, model version, prompt version, revision
              └─ /health
```

- **The revision pins what it serves.** Cloud Build resolves the `production` aliases to version ids at deploy time and sets `MODEL_VERSION_ID` and `EXTRACTOR_VERSION_ID` on the revision. At startup the revision downloads exactly those versions. A revision is immutable, so moving traffic back to an old revision also brings back its model and its prompt.
- **Logging never breaks serving.** A BigQuery failure is logged and the prediction is still returned. Every row carries the model version, so canary and stable can be compared in SQL.
- **The backend is configurable.** `LOG_SINK` = `bigquery` | `stdout` | `none`. `EXTRACT_BACKEND` = `vertex` (Gemini on Vertex AI, ADC, no key) | `gemini-api` (API key) | `ollama` (local model) | `off`. The cloud deployment uses `bigquery` and `vertex`; a laptop can use `stdout` and `ollama`.

## 3. Releases

Every new revision starts at 0% with a tag URL. `canary.sh` puts it at 20% next to the revision that currently has the most traffic, then a person decides: `promote.sh` (100%) or `rollback.sh` (back to stable). The scripts only call `update-traffic`, so no image is rebuilt and the change takes seconds. Two things create revisions: a merge (new code) and a `model-promoted` message (same image, new model version). [Runbook](runbook.md).

## 4. Monitoring loop

```
Cloud Scheduler 06:00 ──▶ Cloud Run Job drift-check
                            ├─ production model = Registry @production
                            ├─ last 24 h of predictions logged by THAT version (BigQuery)
                            ├─ PSI per feature vs. training data, KS on score vs. reference scores
                            └─ drift? ─▶ launcher (guards: no concurrent run, 24 h cooldown) ─▶ pipeline
                                      └▶ Pub/Sub invoice-mlops-drift-alerts (report + retrain result)
```

PSI uses bins cut on the reference quantiles, and repeated edges collapse so that binary and discrete features (`is_new_vendor`, `hour`) work. Thresholds: PSI > 0.25 on any feature, or KS p-value < 0.01 on the score. Below 50 rows the job reports `insufficient_data` instead of a verdict. `scripts/validate_drift.sh` sends normal traffic and then deliberately skewed traffic (amounts ×8, night hours, only new vendors), and checks that only the second batch alerts and starts a retraining run.

## 5. Extraction evaluation loop

`extract-eval` takes a prompt version and runs it on the 40 gold PDFs with the production LLM. It computes precision and recall per field, then micro, and gates on micro precision ≥ 0.85, micro recall ≥ 0.80 and no regression against the prompt version in production. A passing version is registered as a new `invoice-extractor` version (prompt file + full report, label `prompt_version`). A failing one goes to Pub/Sub. The repo ships two prompts so the gate has something to reject: `v1`, a one-line first draft, and `v2`, with normalisation rules for dates, totals, currencies and missing values.

## 6. Identity and infrastructure

Terraform defines every resource ([infra/](../infra/README.md)). Four service accounts, none with a key:

| SA | Can | Cannot |
|---|---|---|
| `invoice-mlops-work` | run the API, jobs and pipeline tasks; read/write its two buckets and the dataset; publish to its topics; use Vertex AI | deploy, change IAM, read anything else |
| `invoice-mlops-build` | build and push images, deploy revisions and move traffic (`run.developer`), publish templates, submit pipelines as `work` | change IAM, delete data |
| `invoice-mlops-ci` | write build logs | everything else: PR builds run code from any branch |
| `invoice-mlops-scheduler` | start the two Cloud Run Jobs | everything else |

The Compute Engine default service account gets project `Editor` by default. No trigger uses it any more, and Terraform removes that grant.

## 7. Why these choices

- **Cloud Run instead of a Vertex AI Endpoint for serving.** A Vertex Endpoint bills at least one node per hour even when it is idle, which on its own costs more than this project's monthly budget. Cloud Run scales to zero, gives traffic splitting and tag URLs for free, and serves a 127 KB XGBoost model fine on CPU. An Endpoint is worth it when you need GPUs, model co-hosting or Vertex's built-in explanations and monitoring. The model is still in the Model Registry, so deploying it to an Endpoint later is one call.
- **Cloud Run Jobs + Cloud Scheduler instead of a Vertex pipeline schedule.** One trigger mechanism for both jobs (drift and training) with guards in code; the launcher logic is unit-tested.
- **Features in SQL.** BigQuery is where the data lives; one SQL file with a pandas twin that is tested for parity. No Feature Store at this scale.
- **Text, not images, to the LLM.** The synthetic PDFs have a text layer, so sending the kept pages' text is cheaper than sending page images. Scanned documents would need OCR or multimodal input.
- **Gemini on the `eu` endpoint.** On 2026-10-22 no GA Gemini Flash model was served from the `europe-west4` regional endpoint. The `eu` multi-region keeps processing in the EU.

## MLOps maturity (Google's levels)

- **Level 0:** model trained and deployed by hand → weeks 1–3.
- **Level 1:** continuous training: parametrised pipeline, scheduled and drift-triggered runs, gate, Registry → weeks 7–8 and 11.
- **Level 2:** CI/CD of the pipeline itself plus monitoring and rollback: PR checks compile and test the pipelines, merges publish the template the scheduled runs use, canary and rollback → weeks 9–11 and 14.
