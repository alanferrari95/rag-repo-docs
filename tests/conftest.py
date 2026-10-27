import pytest

from ragdocs.corpus import Document

RUNBOOK = """# Runbook

Service overview. The API runs on Cloud Run in europe-west4.

## Canary

`canary.sh` sends 20% of traffic to the new revision.

## Rollback

`rollback.sh` moves 100% of traffic back to the previous revision without a rebuild.

```bash
# this is a comment, not a heading
./scripts/rollback.sh
```
"""

DRIFT = """# Drift

## Daily job

The drift job computes PSI and KS on every feature and publishes a Pub/Sub alert when PSI is above 0.2.
"""


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document("demo/runbook.md", "demo", "runbook.md", RUNBOOK),
        Document("demo/drift.md", "demo", "drift.md", DRIFT),
    ]
