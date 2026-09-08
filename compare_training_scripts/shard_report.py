"""Merge the per-rank shard dumps of both revisions and compare them."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path("/home/jovyan/rusakov/runs/compare_training")


def load(side: str, dataset: str):
    directory = ROOT / "artifacts" / f"shard_{dataset}_{side}"
    ranks = sorted(directory.glob("rank*.json"))
    if not ranks:
        raise SystemExit(f"no shard dumps in {directory}")
    payloads = [json.loads(p.read_text()) for p in ranks]
    sets = [set(p["hashes"]) for p in payloads]
    union = set().union(*sets)
    overlap = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap |= sets[i] & sets[j]
    return {
        "side": side,
        "per_rank": [p["n_records"] for p in payloads],
        "per_rank_unique": [len(s) for s in sets],
        "union": len(union),
        "overlap": len(overlap),
        "hashes": union,
    }


def main() -> None:
    dataset = sys.argv[1]
    total = json.loads((ROOT / "artifacts" / f"{dataset}_dims.json").read_text())["train_rows"]
    results = {side: load(side, dataset) for side in ("base", "new")}

    print(f"### shard check — {dataset} (train rows: {total})")
    for side, res in results.items():
        covered = res["union"]
        print(
            f"{side:5} per-rank={res['per_rank']} "
            f"covered={covered} ({covered / total:.1%}) "
            f"overlap={res['overlap']} "
            f"balance={min(res['per_rank']) / max(res['per_rank']):.3f}"
        )

    same = results["base"]["hashes"] == results["new"]["hashes"]
    only_base = len(results["base"]["hashes"] - results["new"]["hashes"])
    only_new = len(results["new"]["hashes"] - results["base"]["hashes"])
    print()
    print(f"identical record coverage across revisions: {same}")
    if not same:
        print(f"  only baseline sees: {only_base}; only refactor sees: {only_new}")

    summary = {
        side: {k: v for k, v in res.items() if k != "hashes"} for side, res in results.items()
    }
    summary["identical_coverage"] = same
    summary["only_base"] = only_base
    summary["only_new"] = only_new
    summary["train_rows"] = total
    (ROOT / "artifacts" / f"shard_{dataset}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten: artifacts/shard_{dataset}.json")


if __name__ == "__main__":
    main()
