# Cost

Target: a portfolio deployment that costs a few euros a month, with a ceiling of €20–30. This page lists what costs money, when it runs, and how to measure it. **No monthly total is published here, because none was measured.** The query below gives the real number once a billing export exists.

## What costs money, and when

| Resource | When it runs | Cost driver | Control in this repo |
|---|---|---|---|
| Cloud Run `invoice-mlops-api` | per request | vCPU-seconds and memory while serving, plus requests | `min-instances=0` (scales to zero; the price is cold starts, see the runbook), max 20 instances |
| Vertex AI Pipelines | each merge (staging), every Monday, after drift | per-run fee plus the CPU of each task | CPU-only tasks, one shared image, caching on staging runs |
| Cloud Run Jobs `drift-check`, `risk-pipeline-launcher` | daily / weekly, seconds to minutes | vCPU-seconds | jobs instead of always-on services |
| Cloud Scheduler | 2 jobs | per job per month beyond the free tier | the free tier covers 3 jobs per billing account |
| Gemini (`/extract`, `extract-eval`) | per document | input and output tokens (thinking tokens are billed as output) | page filter: on the gold set 40 of 117 pages, 47% of the text, reach the model |
| BigQuery | inserts, drift query, feature query | storage and bytes scanned | tiny tables; `extractions` partitioned by day |
| Cloud Storage, Artifact Registry | always | GB stored | no cleanup policy on images on purpose: rollback needs old images |
| Cloud Build | each PR and merge | build minutes beyond the free tier | PR builds only run tests and compile; images are built on merge only |

**Not used on purpose:** Vertex AI Endpoints (a node bills every hour, even idle), Workbench notebooks, GPUs, `min-instances ≥ 1`. In MLOps projects like this one, the line that usually breaks the budget is an Endpoint or a notebook left running.

## Gemini price used by the code

Checked on 2026-10-22 on the [Vertex AI generative AI pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing), Gemini 3.8 Flash, `eu` multi-region ("non-global") rate:

| Period | Input / 1M tokens | Output / 1M tokens |
|---|---|---|
| until 2026-12-31 (launch price) | USD 0.825 | USD 4.125 |
| from 2027-01-01 | USD 1.65 | USD 8.25 |

`src/extract/pricing.py` holds these numbers. `python -m src.eval.evaluate_extract` turns the token counts it measures into `usd_per_1000_docs` for both periods. It has not been run against Gemini, so there is no cost per 1,000 documents yet. With `--backend ollama` the token cost is zero; the cost is the hardware.

## Measuring the month

1. Billing → Billing export → BigQuery export (standard usage cost) into a dataset, e.g. `billing`.
2. Then:

```sql
SELECT service.description AS service,
       ROUND(SUM(cost) + SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)), 2) AS net_cost,
       currency
FROM `<project>.billing.gcp_billing_export_v1_*`
WHERE project.id = '<project>'
  AND invoice.month = FORMAT_DATE('%Y%m', CURRENT_DATE())
GROUP BY service, currency
ORDER BY net_cost DESC;
```

The export only covers usage after it is enabled. For earlier months, use Billing → Reports in the console.

## Budget alert

Create a budget at 50/90/100% of the ceiling in the console. It is scoped to the billing account, not to the project, so it is not in Terraform.
