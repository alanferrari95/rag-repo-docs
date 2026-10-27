# Runbook — invoice-mlops-api

Service: `invoice-mlops-api` · Region: `europe-west4` · Project: `invoice-mlops`
URL: https://invoice-mlops-api-142986012065.europe-west4.run.app

## Release flow

A merge to `main` runs `cloudbuild.yaml`:

1. tests + pipeline compile
2. build and push `api:$SHORT_SHA`
3. `gcloud run deploy --no-traffic --tag=sha-$SHORT_SHA` — the new revision exists but gets 0%
4. `scripts/canary.sh` — new revision at 20%, the current stable revision at 80%

The candidate is also reachable on its own tag URL, without touching the split:
`https://sha-<short-sha>---invoice-mlops-api-142986012065.europe-west4.run.app`

Then a human decides:

| Action | Command | Rebuild? |
|---|---|---|
| See the split | `gcloud run services describe invoice-mlops-api --region=europe-west4 --format='yaml(status.traffic)'` | — |
| Canary at 20% (newest revision) | `./scripts/canary.sh` | no |
| Canary for a given revision / % | `./scripts/canary.sh <revision> 50` | no |
| Promote the canary to 100% | `./scripts/promote.sh` | no |
| Roll back to stable | `./scripts/rollback.sh` (revision with the most traffic) or `./scripts/rollback.sh <revision>` | no |

All three scripts only call `gcloud run services update-traffic`, so they take seconds and never rebuild an image.

## Before promoting

Compare the canary against stable in BigQuery (prediction log, one row per `/score`):

```sql
SELECT model_version, COUNT(*) AS n,
       APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)] AS p95_model_ms,
       AVG(score) AS mean_score
FROM `invoice-mlops.invoice_mlops.predictions`
WHERE ts > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
GROUP BY model_version;
```

End-to-end latency and 5xx per revision come from Cloud Logging (`resource.labels.revision_name`).

## Load test 2026-09-18 (`hey -n 200 -c 10`, POST /score, 80/20 split)

Stable `invoice-mlops-api-00007-7c4` (image `api:8265bec`), canary `invoice-mlops-api-00009-s2m` (same image + env `CANARY=week10`).

Client side (hey): p95 10.3 s, 194 × 200 and 6 × 500, caused by cold starts (`min-instances=0`).

Per revision (Cloud Logging):

| revision | n | 5xx | err% | p50 | p95 |
|---|---|---|---|---|---|
| 00007-7c4 | 157 | 6 | 3.8% | 0.277 s | 0.659 s |
| 00009-s2m | 43 | 0 | 0.0% | 0.105 s | 0.323 s |

## Known pitfall (fixed)

`rollback.sh` used to pin 100% of traffic to a hard-coded revision. While traffic is pinned to a named revision, `gcloud run deploy` creates new revisions at 0%, so the merges after 2026-09-18 (revisions 00010 and 00011) never received traffic. Deploys now explicitly use `--no-traffic` + `canary.sh`, so the pin is intentional and visible, and rollback takes the stable revision from the live split instead of a constant.

## Do not

- Do not create Vertex AI Endpoints (billed per hour even when idle).
- Do not delete the Cloud Run service, buckets, Artifact Registry repo or the BigQuery dataset: they are managed by Terraform (`infra/`).
