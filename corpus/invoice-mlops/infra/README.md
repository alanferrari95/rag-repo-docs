# infra/

Terraform for the whole GCP side. Works in any project: everything is derived from `project_id`. State lives in a versioned GCS bucket (`<project_id>-tfstate`).

| File | What |
|---|---|
| `versions.tf` | Terraform/provider versions, GCS backend |
| `variables.tf` | project, region, GitHub repo, API min instances, Scheduler region |
| `main.tf` | APIs, buckets, Artifact Registry, work service account |
| `data.tf` | BigQuery dataset, `invoices`, `predictions`, `extractions` |
| `serving.tf` | Cloud Run service (identity, scaling, public access) |
| `monitoring.tf` | Pub/Sub topics, drift + launcher Cloud Run Jobs, Cloud Scheduler |
| `iam.tf` | least-privilege grants for the four service accounts |
| `cicd.tf` | Cloud Build triggers (PR, main, model promotion) |
| `examples/` | imports for the original project, where some resources predate Terraform |

## Check it offline (no GCP account)

```bash
terraform -chdir=infra fmt -check -recursive
terraform -chdir=infra init -backend=false
terraform -chdir=infra validate
```

The PR build runs the same three commands.

## Bring it up in your own project

```bash
gcloud auth login && gcloud auth application-default login   # no service-account keys
./infra/bootstrap.sh <project_id>                              # state bucket, once
cp infra/terraform.tfvars.example infra/terraform.tfvars       # set project_id, github_owner
terraform -chdir=infra init -backend-config=bucket=<project_id>-tfstate
terraform -chdir=infra plan
terraform -chdir=infra apply
```

Then, once:

1. Connect the GitHub repo to Cloud Build (GitHub App, one click in the console; 1st-gen triggers cannot do this from Terraform).
2. Build the images: `gcloud builds submit --config=cloudbuild-images.yaml --substitutions=_TAG=$(git rev-parse --short HEAD)`.
3. Seed BigQuery: `./scripts/seed_bigquery.sh`.
4. First models: `python -m src.launcher --trigger manual` (risk model → Registry `production`) and `python pipelines/submit_extract_eval.py --prompt-version v2` (prompt → Registry `production`).
5. Merge or push to `main`: the build deploys the API as a 20% canary pinned to those two versions; `./scripts/promote.sh` sends it 100%.

Until step 4 has run, merges build and test normally and skip the deploy with a message.

## Cost controls built in

`min-instances=0` on Cloud Run, Cloud Run Jobs instead of always-on services, no Vertex AI Endpoints (Cloud Run serves the model), CPU-only pipeline components. Two Cloud Scheduler jobs fit in the free tier (3 per billing account). Set a billing budget with alerts in the console; it is billing-account scoped, so it is not in this config.

## Not in Terraform on purpose

The state bucket (chicken-and-egg), the billing budget, and what Cloud Build owns at runtime: images, the traffic split, Registry versions.
