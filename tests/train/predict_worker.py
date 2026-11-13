"""One rank of a multi-rank :func:`avatar.train.evaluate.predict` run.

Built as a separate script because the thing under test only exists across
processes: whether the ranks agree on how many collectives they perform, and
whether two of them writing into one directory keep each other's files.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())

from avatar.data.tabular import TabularCollateFn, TabularDataset
from avatar.metrics.base import ArtifactMetric, ScalarMetric
from avatar.train.dist import DistEnv
from avatar.train.evaluate import predict


@dataclasses.dataclass
class Output:
    """Minimal dataclass output — ``narrow_for_metrics`` needs the fields."""

    score: torch.Tensor | None = None
    loss: torch.Tensor | None = None


class Identity(torch.nn.Module):
    """Score every record with its own row index; the value does not matter."""

    def forward(self, tab_features=None, **kwargs):
        size = tab_features.cat_features.shape[0]
        return Output(score=torch.arange(size, dtype=torch.float))


class CollectPopulation(ScalarMetric):
    """Needs every record on one rank, like ROC-AUC or Qini."""

    required_inputs = ("id",)
    required_outputs = ("score",)

    def __init__(self):
        self.ids: list[int] = []

    def update(self, inputs, outputs):
        self.ids.extend(int(value) for value in inputs["id"])

    def compute(self):
        return {"ids": sorted(self.ids)}

    def reset(self):
        pass  # the test reads them after predict returns


class DumpScores(ArtifactMetric):
    """Writes rows to a directory; no rank needs another rank's rows."""

    required_inputs = ("id",)
    required_outputs = ("score",)

    def __init__(self, path_to_save):
        super().__init__(path_to_save=path_to_save)
        self.rows: list[dict] = []

    def update(self, inputs, outputs):
        self.rows.append({"id": [int(value) for value in inputs["id"]]})

    def flush(self):
        if not self.rows:
            return
        self.save_dataframe(
            pd.DataFrame({"id": [i for row in self.rows for i in row["id"]]})
        )
        self.rows = []

    def reset(self):
        self.rows = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--metric", choices=["population", "artifact"], required=True)
    parser.add_argument("--drop-tail", dest="drop_tail", action="store_true")
    parser.set_defaults(drop_tail=False)
    args = parser.parse_args()

    env = DistEnv.from_env(timeout_sec=120)
    dataset = TabularDataset(
        path=args.path,
        shuffle_pq=False,
        drop_tail=args.drop_tail,
        filter_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=TabularCollateFn(),
        num_workers=0,
    )

    metric = (
        CollectPopulation()
        if args.metric == "population"
        else DumpScores(args.dump_dir)
    )
    scores = predict(Identity(), loader, env, metric)

    report = {
        "rank": env.rank,
        "world_size": env.world_size,
        "records_owned": dataset.planner.rank_record_count(),
        "total_records": dataset.planner.total,
        "scored_ids": (scores or {}).get("ids"),
    }
    with open(os.path.join(args.out, f"rank{env.rank}.json"), "w") as handle:
        json.dump(report, handle)
    env.destroy()


if __name__ == "__main__":
    main()
