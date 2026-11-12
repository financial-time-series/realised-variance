"""Pull every checkpoint into the project cache. LOGIN NODE ONLY.

    ssh <REMOTE_USER>@<REMOTE_HOST> 'cd <REMOTE_DIR> && export PATH=$HOME/.local/bin:$PATH \
        && HF_HOME=$PWD/.hf_cache uv run --no-sync python scripts/prefetch.py'

Compute nodes have no route to the internet; the login node does. Without this
step every foundation-model job spends its first minutes retrying a download it
can never complete and then dies with a message that reads like a transient
outage. `slurm/run.slurm` sets HF_HUB_OFFLINE=1 so the failure is instant and
honest instead.

Also re-reads each checkpoint's creation date from the Hub and checks it
against the date hardcoded in `finlab.zoo`. Those dates decide where the
leakage-controlled evaluation window starts, so a silent drift between what the
registry claims and what the Hub says would quietly invalidate the paper's main
control.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finlab.zoo import GENERATION_WINDOWS, ZOO, check_release_dates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=sorted(ZOO))
    ap.add_argument("--verify-dates", action="store_true", default=True)
    args = ap.parse_args()

    if "HF_HOME" not in os.environ:
        print("WARNING: HF_HOME is unset -- weights will land in ~/.cache, "
              "which is quota'd. Set HF_HOME=$PWD/.hf_cache.")

    check_release_dates()
    for gen, start in sorted(GENERATION_WINDOWS.items()):
        members = [s for s in ZOO.values() if s.generation == gen]
        if members:
            print(f"{gen}: window opens {start}, newest member released "
                  f"{max(s.released for s in members)} "
                  f"({len(members)} checkpoints)")
    print()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    failed, drifted = [], []
    for name in args.models:
        spec = ZOO[name]
        print(f"--- {name:<20} {spec.repo}")
        try:
            # `main`, so the cache gets refs/main and offline loaders find it;
            # then the resolved commit must equal the pinned one, or the study
            # would silently run different weights from the ones it reports.
            path = snapshot_download(spec.repo, revision="main")
            sha = Path(path).name
            size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
            print(f"    cached ({size / 1e6:.0f} MB) at {sha[:12]}")
            if spec.revision != "main" and sha != spec.revision:
                print(f"    !!! revision drift: registry pins {spec.revision[:12]}, "
                      f"main is now {sha[:12]}")
                drifted.append((name, spec.revision[:12], sha[:12]))
        except Exception as exc:  # noqa: BLE001 - one model must not stop the rest
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            failed.append(name)
            continue

        if args.verify_dates:
            try:
                created = str(api.model_info(spec.repo).created_at)[:10]
                if created != spec.released:
                    print(f"    !!! release date drift: zoo says {spec.released}, "
                          f"Hub says {created}")
                    drifted.append((name, spec.released, created))
                else:
                    print(f"    created {created} -- matches registry")
                # The date the loaded weight/config files last changed. A
                # checkpoint whose weights were replaced after its repository
                # was created became public, as used here, on the later date.
                tree = api.list_repo_tree(spec.repo, revision=spec.revision,
                                          recursive=True, expand=True)
                dates = [str(f.last_commit.date)[:10] for f in tree
                         if getattr(f, "last_commit", None) is not None
                         and f.path.endswith((".safetensors", ".bin", "config.json"))]
                changed = max(dates) if dates else None
                if spec.files_changed and changed != spec.files_changed:
                    print(f"    !!! file date drift: zoo says {spec.files_changed}, "
                          f"Hub says {changed}")
                    drifted.append((name, spec.files_changed, changed))
                else:
                    print(f"    files last changed {changed} -- matches registry; "
                          f"evaluated from {spec.effective_release}")
            except Exception as exc:  # noqa: BLE001
                print(f"    (date check skipped: {type(exc).__name__}: {exc})")

    print()
    if drifted:
        print("!!! DRIFT -- the leakage window or the weights may be wrong:")
        for name, claimed, actual in drifted:
            print(f"    {name}: registry {claimed}, hub {actual}")
        return 2
    if failed:
        print(f"!!! failed to fetch: {failed}")
        return 1
    print(f"all {len(args.models)} checkpoints cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
