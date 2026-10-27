# Personal data (PII) note

**Rule: this project never processes real customer data.** Every invoice, vendor, customer, IBAN and tax id in the repository, in the buckets and in BigQuery is synthetic (Faker + rule-generated values, `src/features/generate_dataset.py`, `src/extract/synthetic_docs.py`).

## Why it matters even for a demo

Invoices are full of personal and commercial data: names of sole traders, addresses, bank accounts, tax ids, amounts. Under GDPR each of those needs a legal basis, a retention period and a processor agreement with every service that sees it, including the LLM provider. A portfolio project has none of that, so the only safe input is invented data.

## What the system does with what it receives

| Where | What is stored | What is never stored |
|---|---|---|
| `predictions` (BigQuery) | the 6 numeric features, score, model version, latency | vendor or customer names, invoice ids |
| `extractions` (BigQuery) | SHA-256 of the PDF, page counts, token counts, latency, model and prompt version, number of fields found | the PDF, its text, the extracted values |
| Cloud Logging | request metadata, errors (exception type only for Gemini failures) | request bodies |
| Gemini | the text of the pages kept by the page filter, for one request | — (Vertex AI does not use customer data to train models; the `eu` endpoint keeps processing in the EU) |

The page filter is also a data-minimisation control: cover letters and remittance slips (bank details, contact names) are not sent to the LLM when they are not needed.

## If this ever handled real documents

1. Data Processing Agreement with Google Cloud, a record of processing, and a DPIA for the LLM step.
2. Run Cloud DLP (Sensitive Data Protection) on uploads, and keep raw documents in a separate bucket with a short lifecycle rule (e.g. delete after 30 days) and CMEK.
3. Put the API behind authentication (IAP or ID tokens) instead of the public demo endpoint, and restrict ingress.
4. VPC Service Controls around BigQuery, Cloud Storage and Vertex AI to prevent exfiltration.
5. Keep the gold set synthetic or fully anonymised; never copy production documents into `data/gold/`.

## Secrets

No secrets in git (`.gitignore` blocks `.env`, key files and Terraform state). No service-account JSON keys exist: Cloud Run, Cloud Run Jobs, Cloud Build and Vertex AI get short-lived tokens for their own service account, and humans use `gcloud auth login`.
