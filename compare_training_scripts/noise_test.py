"""Is the difference between revisions a bias, or just seed noise?

Two questions, both answered from the finished runs:

1. **Magnitude.** For each metric, compare the differences between revisions at
   equal seed (|base(s) - new(s)|) with the differences the seed alone makes
   inside the baseline (|base(s_i) - base(s_j)| over all seed pairs). If the
   first set sits inside the range of the second, the revisions are
   indistinguishable at this sample size.

2. **Direction.** A real change of behaviour pushes a metric the same way at
   every seed. Mixed signs across seeds mean noise. With four seeds, all four
   agreeing by chance has probability 1/8.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path("/home/jovyan/rusakov/runs/compare_training")

HEADLINE = {
    "valid_avg_calib_roc_auc_score",
    "valid_avg_calib_treatment_roc_auc_score",
    "valid_avg_calib_control_roc_auc_score",
    "valid_avg_calib_qini_auc_score",
    "valid_avg_calib_uplift_at_20",
    "valid_mean_qini_auc_score",
    "train_loss",
}


def main() -> None:
    dataset = sys.argv[1]
    data = json.loads((ROOT / f"artifacts/metrics_{dataset}.json").read_text())

    rows = []
    for key, entry in data.items():
        base = entry["base_finals"]
        new = entry["new_finals"]
        if len(base) != len(new) or len(base) < 2:
            continue
        cross = [abs(a - b) for a, b in itertools.combinations(base, 2)]
        paired = [abs(a - b) for a, b in zip(base, new)]
        signed = [n - b for b, n in zip(base, new)]
        same_sign = all(s > 0 for s in signed) or all(s < 0 for s in signed)
        rows.append({
            "metric": key,
            "paired_max": max(paired),
            "cross_max": max(cross),
            "paired_mean": statistics.fmean(paired),
            "cross_mean": statistics.fmean(cross),
            "inside": max(paired) <= max(cross),
            "same_sign": same_sign,
            "signed": signed,
        })

    n = len(rows)
    inside = sum(r["inside"] for r in rows)
    biased = [r for r in rows if r["same_sign"] and not r["inside"]]

    print(f"### {dataset}: {n} metrics, {len(data[next(iter(data))]['base_finals'])} seeds per side")
    print()
    print(f"{'metric':46} {'paired max':>11} {'seed max':>10} {'paired mean':>12} {'seed mean':>10}  direction")
    print("-" * 104)
    for row in sorted(rows, key=lambda r: (r["metric"] not in HEADLINE, r["metric"])):
        mark = "same sign" if row["same_sign"] else "mixed"
        print(
            f"{row['metric']:46} {row['paired_max']:>11.5f} {row['cross_max']:>10.5f} "
            f"{row['paired_mean']:>12.5f} {row['cross_mean']:>10.5f}  {mark}"
        )

    print()
    print(f"paired difference inside the seed-to-seed range: {inside}/{n}")
    print(f"metrics that are both outside that range and consistent in direction: {len(biased)}")
    for row in biased:
        print(f"  {row['metric']}: signed diffs {[round(s, 5) for s in row['signed']]}")


if __name__ == "__main__":
    main()
