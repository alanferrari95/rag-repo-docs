"""Copy the documents listed in corpus/manifest.yaml into corpus/<repo>/.

Each file is read with `git show <commit>:<path>` from a local clone, so the
snapshot is exactly what was committed at the pinned commit, not whatever is
in that clone's working tree today.

    python scripts/snapshot_corpus.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"


def main() -> int:
    manifest = yaml.safe_load((CORPUS / "manifest.yaml").read_text(encoding="utf-8"))
    total_words = 0
    for repo in manifest["repos"]:
        clone = (ROOT / repo["local"]).resolve()
        for rel in repo["paths"]:
            text = subprocess.run(
                ["git", "-C", str(clone), "show", f"{repo['commit']}:{rel}"],
                capture_output=True, check=True, encoding="utf-8",
            ).stdout
            out = CORPUS / repo["name"] / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8", newline="\n")
            words = len(text.split())
            total_words += words
            print(f"{words:6d}  {repo['name']}/{rel}")
    print(f"{total_words:6d}  total words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
