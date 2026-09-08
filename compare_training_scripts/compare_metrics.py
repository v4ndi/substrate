"""Compare the metric curves of the two revisions across several seeds.

The two sides cannot be bit-equal — the dataloader, the RNG consumption and the
distributed wrapper were all rewritten — so the question is whether the
difference between revisions is larger than the difference a seed change makes
inside one revision.

Two readings are printed:

* **paired** — same seed, both revisions: |base(s) - new(s)| per metric;
* **spread** — inside the baseline: the range across its own seeds.

A metric whose paired difference stays inside the baseline's own seed range is
indistinguishable from noise. One that does not is a real behavioural change.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

import mlflow

ROOT = pathlib.Path("/home/jovyan/rusakov/runs/compare_training")
mlflow.set_tracking_uri(f"file:{ROOT}/mlruns")

HEADLINE = [
    "valid_avg_calib_roc_auc_score",
    "valid_avg_calib_qini_auc_score",
    "valid_avg_calib_uplift_at_20",
    "valid_avg_calib_treatment_roc_auc_score",
    "valid_avg_calib_control_roc_auc_score",
    "valid_avg_test_calibrated_qini_auc_score",
    "valid_mean_qini_auc_score",
    "train_loss",
]

CLIENT = mlflow.MlflowClient()


def load(dataset: str, tag: str):
    experiment = CLIENT.get_experiment_by_name(f"compare_{dataset}")
    runs = [
        r
        for r in CLIENT.search_runs(
            [experiment.experiment_id],
            filter_string=f"attributes.run_name = '{tag}'",
            order_by=["attributes.start_time DESC"],
        )
        if r.info.status == "FINISHED"
    ]
    if not runs:
        return None
    run = runs[0]
    history = {}
    for key in run.data.metrics:
        points = CLIENT.get_metric_history(run.info.run_id, key)
        history[key] = [
            p.value for p in sorted(points, key=lambda p: (p.step, p.timestamp))
        ]
    return history


def final(history, key):
    values = history.get(key) or []
    return values[-1] if values else None


def main() -> None:
    dataset = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [42, 43]

    runs = {}
    for side in ("base", "new"):
        for seed in seeds:
            tag = f"{dataset}_{side}_seed{seed}"
            history = load(dataset, tag)
            if history is None:
                print(f"!! missing run {tag}")
                continue
            runs[(side, seed)] = history

    have = sorted({seed for (side, seed) in runs if (("base", seed) in runs and ("new", seed) in runs)})
    base_seeds = sorted({seed for (side, seed) in runs if side == "base"})
    print(f"### {dataset}")
    print(f"paired seeds: {have}; baseline seeds: {base_seeds}")
    epochs = {tag: len(hist.get("train_loss", [])) for tag, hist in runs.items()}
    print(f"epochs logged: {sorted(set(epochs.values()))}\n")

    keys = [k for k in HEADLINE if any(k in hist for hist in runs.values())]
    extra = sorted(
        k
        for k in set().union(*[set(h) for h in runs.values()])
        if k.startswith("valid_") and k not in keys
    )

    report = {}
    header = (
        f"{'metric':44} {'base(mean)':>11} {'new(mean)':>11} "
        f"{'paired|Δ|max':>13} {'seed spread':>12}  verdict"
    )
    print(header)
    print("-" * len(header))

    exceeded = []
    for key in keys + extra:
        base_finals = [
            final(runs[("base", s)], key) for s in base_seeds if ("base", s) in runs
        ]
        base_finals = [v for v in base_finals if v is not None]
        new_finals = [
            final(runs[("new", s)], key) for s in have if ("new", s) in runs
        ]
        new_finals = [v for v in new_finals if v is not None]
        paired = []
        for seed in have:
            a, b = final(runs[("base", seed)], key), final(runs[("new", seed)], key)
            if a is not None and b is not None:
                paired.append(abs(a - b))
        if not base_finals or not new_finals or not paired:
            continue
        # The valid set carries no "test" slice (there is no split_type column),
        # so every test_* average is a constant zero on both sides. Nothing to
        # compare there.
        if not any(base_finals) and not any(new_finals):
            continue

        spread = max(base_finals) - min(base_finals) if len(base_finals) > 1 else 0.0
        worst = max(paired)
        ok = worst <= spread if len(base_finals) > 1 else None
        if ok is False:
            exceeded.append((key, worst, spread))
        report[key] = {
            "base_finals": base_finals,
            "new_finals": new_finals,
            "paired_abs_diffs": paired,
            "baseline_seed_spread": spread,
            "within_seed_spread": ok,
        }
        verdict = {True: "within seed spread", False: "EXCEEDS seed spread", None: "n/a"}[ok]
        print(
            f"{key:44} {statistics.fmean(base_finals):>11.5f} "
            f"{statistics.fmean(new_finals):>11.5f} {worst:>13.5f} {spread:>12.5f}  {verdict}"
        )

    total = len(report)
    inside = sum(1 for v in report.values() if v["within_seed_spread"])
    print()
    print(f"metrics compared: {total}; within baseline seed spread: {inside}; exceeding: {len(exceeded)}")
    for key, worst, spread in sorted(exceeded, key=lambda item: item[1] - item[2], reverse=True)[:10]:
        print(f"  {key}: paired|Δ|={worst:.5f} vs seed spread {spread:.5f}")

    out = ROOT / "artifacts" / f"metrics_{dataset}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
