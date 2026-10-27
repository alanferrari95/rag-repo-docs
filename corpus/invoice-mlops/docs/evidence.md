# Evidence

What backs each claim about this project, and how strong that backing is. Cloud facts come from read-only commands against the reference deployment (`invoice-mlops`, `europe-west4`) on 2026-10-22; IDs are copied from command output.

Three levels of evidence:

- **Ran on GCP:** it happened in the reference project and left a build, run, job or table row behind.
- **Tested:** the code exists and unit tests exercise it with fakes, but it has not run against the cloud.
- **Code only:** written and reviewed, not executed (Terraform plan/apply, the drift validation script).

## CV bullet → files and evidence

| # | CV bullet | Files | Evidence | Level |
|---|---|---|---|---|
| 1 | Vertex AI Pipelines (KFP v2) with continuous training: BigQuery features, containerized XGBoost training, evaluation against a gold set and a promotion gate into Vertex AI Model Registry; weekly runs from Cloud Scheduler. | `pipelines/risk_train_pipeline.py`, `src/features/features.sql`, `src/train/train.py`, `src/eval/gate.py`, `src/registry.py`, `src/launcher.py`, `infra/monitoring.tf`, `data/gold/risk_gold.csv` | Sep 2026: KFP train → metric gate runs on Vertex (`risk-train-eval` 09-09 ✓, `risk-train-eval-week8-fail` 09-10 ✗, staging 09-15/25/27 ✓), containerized training (`train:week4`), 3 Vertex Training custom jobs. New graph (BigQuery features, gold gate vs. champion, Registry promotion) and Cloud Scheduler: PRs [#8](https://github.com/alanferrari95/invoice-mlops/pull/8), [#9](https://github.com/alanferrari95/invoice-mlops/pull/9), tests `test_pipeline_compiles.py`, `test_gate.py`, `test_build_features.py`, `test_launcher.py`. | Partly ran on GCP (previous pipeline graph); new graph **tested** |
| 2 | CI/CD/CT with Cloud Build: tests and pipeline compilation on every PR; build, push, Cloud Run deploy and a staging pipeline run on merge. | `cloudbuild-pr.yaml`, `cloudbuild.yaml`, `pipelines/submit_staging.py` | Green PR builds for every PR in this stack (#5 `33041b6f…`, #6 `d9dc1ebf…`, #7 `f8e51aa0…`, #8 `b61eb571…`, #9 `b8298293…`, #10 `85a83bb4…`, #11 `eba29268…`); red PR on a deliberate schema bug `07f7309b…` (commit `b2c3cc1`); merge builds that deployed and submitted staging runs, e.g. `9c4e47a1…` (commit `e637a1e`). | **Ran on GCP** |
| 3 | FastAPI on Cloud Run (/score, /extract, /health, /version) with Pandera input validation; every prediction, latency and token count logged to BigQuery. | `src/api/main.py`, `infra/data.tf` | `/score`, `/health`, `/version` served from Cloud Run since Sep; 395 rows in `invoice_mlops.predictions` (2026-09-04 → 10-13). `/extract` and the `extractions` table (tokens, latency): [#10](https://github.com/alanferrari95/invoice-mlops/pull/10), tests `test_extract_api.py`, `test_score_validation.py`. | /score **ran on GCP**; /extract **tested** |
| 4 | Canary releases at 20% traffic and one-command rollback without rebuild; load-tested at p95 of 0.32–0.66 s per revision (warm). | `scripts/canary.sh`, `promote.sh`, `rollback.sh`, `docs/runbook.md` | 2026-09-18: 80/20 split between `00007-7c4` and `00009-s2m`, `hey -n 200 -c 10`, p95 0.659 s / 0.323 s per revision from Cloud Logging, rollback to 100% `00007-7c4`. Release-flow fix: [#7](https://github.com/alanferrari95/invoice-mlops/pull/7). | **Ran on GCP** |
| 5 | Daily drift job (PSI per feature, KS test on the score) that publishes Pub/Sub alerts and re-triggers the training pipeline; validated with deliberately skewed traffic. | `src/monitoring/drift.py`, `src/launcher.py`, `scripts/send_traffic.py`, `scripts/validate_drift.sh`, `infra/monitoring.tf` | Topic `invoice-mlops-drift-alerts` + subscription exist. Skewed-traffic validation in `tests/test_drift_job.py` (normal → ok, skewed → alert + retrain) and `tests/test_drift.py`. [#5](https://github.com/alanferrari95/invoice-mlops/pull/5), [#9](https://github.com/alanferrari95/invoice-mlops/pull/9). | **Tested** |
| 6 | Gemini extraction module with structured output and page filtering before the LLM call, evaluated with per-field precision and recall on a hand-labeled gold set; a pipeline gate rejects worse prompt versions. | `src/extract/`, `src/eval/extract_metrics.py`, `src/eval/evaluate_extract.py`, `pipelines/extract_eval_pipeline.py`, `data/gold/extract/` | Page filter keeps exactly the invoice page on 40/40 gold documents (`test_extract.py`, `scripts/local_report.py`); per-field P/R and gate logic tested (`test_extract_metrics.py`); prompt gate pipeline compiles (`test_pipeline_compiles.py`). [#10](https://github.com/alanferrari95/invoice-mlops/pull/10), [#12](https://github.com/alanferrari95/invoice-mlops/pull/12). No run against Gemini. Labels come from the generator, not hand-labelled. | **Tested** |
| 7 | Everything in Terraform (buckets, Artifact Registry, Cloud Run, IAM, Cloud Build triggers) with a least-privilege service account and no JSON keys; SHAP explanations for the risk model. | `infra/*.tf`, `scripts/shap_report.py`, `docs/img/shap_*.png`, `src/api/main.py` (`?explain=true`) | Terraform code for all of it: [#11](https://github.com/alanferrari95/invoice-mlops/pull/11), [#12](https://github.com/alanferrari95/invoice-mlops/pull/12). No JSON keys: `gcloud iam service-accounts keys list --managed-by=user` → 0. Week-0 buckets and SA were created by Terraform. SHAP plots in `docs/img/`; per-request SHAP tested. | Buckets/SA **ran on GCP**; rest **code only** (`terraform validate` and `fmt` pass, also checked in the PR build); SHAP **ran locally** |

## Graduation checklist (study plan)

| Item | Status |
|---|---|
| `terraform apply` documented and tested on an empty project | Documented (`infra/README.md`); `init -backend=false`, `validate` and `fmt` pass (Terraform 1.17.0, google provider 6.50.0, 2026-10-22); `plan`/`apply` **not run** |
| Pipeline on Vertex with visible lineage | Earlier graph ran on Vertex; the new graph passes artifacts end to end (features → model → registered version), not run yet |
| Model Registry versioned (classifier + extraction prompt version) | Code for both (`risk-classifier`, `invoice-extractor`); the Registry holds one manually uploaded classifier version |
| Cloud Run: `/score` `/extract` `/version` `/health`, canary 20% and rollback demonstrated | `/score` `/version` `/health` and canary/rollback demonstrated; `/extract` tested only |
| Input validation active (rejects corrupt input) | ✅ (Pandera; 422 tests now run in CI) |
| Load test with documented p95 | ✅ (runbook, 2026-09-18) |
| Field precision on the gold set ≥ the threshold you set | Threshold set (micro P ≥ 0.85); not measured against an LLM |
| Drift alert fired on purpose | Fired in tests with skewed data; not in the cloud |
| `docs/cost.md` with real € of the month | Cost drivers and the measuring query are documented; the month's € was not recorded |
| README a hiring manager understands in 3 minutes | ✅ |
| Zero secrets in git | ✅ (`.gitignore`, no keys exist, `.env.example` only) |

## State of the reference project on 2026-10-22

| What | Evidence | Command |
|---|---|---|
| Vertex AI Training custom jobs (week 4) | `risk-classifier-custom-job` (2026-09-02), `risk-train-n200-d3-custom-job`, `risk-train-n50-d6-custom-job` (2026-09-03), all `JOB_STATE_SUCCEEDED`; artifacts in `gs://invoice-mlops-artifacts/models/exp-n{100,200,50}-d{4,3,6}/` | `gcloud ai custom-jobs list --region=europe-west4` |
| Vertex AI Experiments | none recorded | `aiplatform.Experiment.list()` |
| Model Registry | `risk-classifier` (id `429981614227849216`), 1 version, alias `default`, artifact `gs://invoice-mlops-artifacts/models/0.2.1-xgb21-bst/` | `gcloud ai models list-version 429981614227849216 --region=europe-west4` |
| Pipeline runs | `risk-train-eval` 2026-09-09 SUCCEEDED; `risk-train-eval-week8-fail` 2026-09-10 FAILED; 5 runs from a Vertex schedule on 2026-09-10; staging runs 2026-09-15, 09-25, 09-27 SUCCEEDED | `aiplatform.PipelineJob.list()` |
| Weekly trigger | Vertex AI Pipelines schedule `risk-train-eval-scheduled` (`0 8 * * 1`, Europe/Madrid), **paused**; Cloud Scheduler API not enabled | `aiplatform.PipelineJobSchedule.list()`, `gcloud scheduler jobs list` |
| CI/CD | triggers `test-pr` (PRs to `main`) and `deploy-main` (push to `main`), both running as the Compute Engine default SA | `gcloud builds triggers list --region=europe-west4` |
| Cloud Run traffic | 100% on `invoice-mlops-api-00007-7c4` (image `api:8265bec`); revisions 00010 and 00011 deployed at 0% because the week-10 rollback pinned traffic | `gcloud run services describe invoice-mlops-api --format='yaml(status.traffic)'` |
| Cloud Run config | runs as `invoice-mlops-work`, public via `invoker_iam_disabled`, max 20 instances, 512 MiB | Cloud Run Admin API |
| Prediction log | 395 rows, all `model_version = 0.1.0`, 2026-09-04 → 2026-10-13 | BigQuery |
| Least-privilege gap | Compute Engine default SA has `roles/editor` and runs the triggers | `gcloud projects get-iam-policy invoice-mlops` |
| Drift plumbing | topic `invoice-mlops-drift-alerts` (no publisher grant yet) + subscription `drift-alerts-debug` | Pub/Sub API |

## Turning "tested" into "ran on GCP"

In a project brought up with `infra/README.md`:

| Evidence | Command | Produces |
|---|---|---|
| First production model + lineage | `python -m src.launcher --trigger manual` | pipeline run, Registry version with alias `production` |
| Prompt gate accepts v2, rejects v1 | `python pipelines/submit_extract_eval.py --prompt-version v2`, then `v1` | two runs, one `invoice-extractor` version, one Pub/Sub rejection |
| Per-field P/R, tokens, cost / 1,000 docs | `python -m src.eval.evaluate_extract --backend vertex --out report.json` | numbers for the README |
| Drift alert on purpose | `./scripts/validate_drift.sh` | job logs, Pub/Sub message, pipeline run with `trigger=drift` |
| Scheduler | `gcloud scheduler jobs run risk-pipeline-weekly --location=europe-west4` | pipeline run with `trigger=schedule` |
| QPS / p95 of the new revision | `hey -n 200 -c 10 -m POST -T application/json -D payload.json <url>/score` | load-test table |
