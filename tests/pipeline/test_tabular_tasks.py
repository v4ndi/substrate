"""The four tabular settings, each from a record to a number.

Uplift, response, regression and multi-class differ in the collate function,
the head and the metric, and every defect these tests pin down lived in a joint
between those three rather than inside any one of them. So each test walks the
whole path a training step walks: records -> collate -> pipeline -> loss ->
backward -> metric.update -> metric.compute.
"""

from __future__ import annotations

import numpy as np
import torch

from avatar.data import SupervisedCollateFn, UpliftCollateFn
from avatar.metrics import (
    MultiClassMetrics,
    RegressionMetrics,
    ResponseMetrics,
    UpliftMetrics,
)
from avatar.nn.embedding import TabularEmbedding
from avatar.nn.tabular import TabularTransformer
from avatar.pipeline.tabular import (
    SupervisedLearner,
    TabularClassification,
    TabularWithAggregatedStates,
)
from avatar.pipeline.uplift import SLearner

RECORDS = 64
N_CAT, N_NUM, WIDTH, VOCAB = 3, 5, 16, 40
N_GROUPS = 2


def records(target_of):
    """One epoch's worth of records, as ``TabularDataset`` would yield them.

    Group, split, treatment arm and class cycle at different rates — 2, 4, 8 and
    16 records — so that each of the sixteen combinations holds four records.
    Tie any two of them to the same parity and every slice ends up with one
    class in it, which is a metric that cannot be computed rather than a test.
    """
    rng = np.random.default_rng(0)
    out = []
    for index in range(RECORDS):
        out.append({
            "epk_id": index,
            "target": target_of(index),
            "treatment": (index // 4) % 2,
            # The column is called ``group`` because that is the name the
            # metrics require; both collate functions used to delete it.
            "group": index % N_GROUPS,
            "split_type": "calib" if (index // 2) % 2 else "test",
            "tab_features": {
                "cat_features": torch.tensor(rng.integers(0, VOCAB, N_CAT)),
                "num_features": torch.tensor(
                    rng.standard_normal(N_NUM), dtype=torch.float32
                ),
            },
        })
    return out


def class_of(classes: int):
    """A class label that cycles slower than group, split and treatment do."""
    return lambda index: (index // 8) % classes


def representation(extra_tokens: int = 0):
    return TabularWithAggregatedStates(
        embedding=TabularEmbedding(
            num_numerical_features=N_NUM, hidden_size=WIDTH, vocab_size=VOCAB
        ),
        encoder=TabularTransformer(hidden_size=WIDTH, num_heads=2, num_layers=1),
        aggregation_config={
            "name": "linear",
            "num_features": N_CAT + N_NUM + extra_tokens,
            "emb_dim": WIDTH,
        },
    )


def one_step(model, batch, metric):
    """Train one step on the batch, then score it. Returns the metric's numbers."""
    model.train()
    output = model(**batch)
    output.loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    ), "the loss reached no parameter"

    model.eval()
    with torch.no_grad():
        output = model(**batch)
    metric.update(inputs=batch, outputs=output)
    return metric.compute()


def test_response_reaches_a_ranking_number():
    batch = SupervisedCollateFn(
        target_column="target", add_extra_columns={"group": "group"}
    )(records(class_of(2)))

    scores = one_step(
        TabularClassification(
            num_classes=1, tabular_model=representation(), task_type="classification"
        ),
        batch,
        ResponseMetrics(),
    )

    assert 0.0 <= scores["mean_roc_auc_score"] <= 1.0
    assert "calib_group_0_roc_auc_score" in scores
    assert "test_group_1_roc_auc_score" in scores


def test_regression_reaches_an_error_number():
    batch = SupervisedCollateFn(
        target_column="target",
        is_regression=True,
        add_extra_columns={"group": "group"},
    )(records(lambda index: float(index) / RECORDS))

    scores = one_step(
        TabularClassification(
            num_classes=1, tabular_model=representation(), task_type="regression"
        ),
        batch,
        RegressionMetrics(),
    )

    assert scores["mean_mae"] > 0
    assert {"calib_group_0_mae", "calib_group_0_rmse", "calib_group_0_r2"} <= set(
        scores
    )


def test_multiclass_reaches_a_balanced_number():
    classes = 4
    batch = SupervisedCollateFn(
        target_column="target", add_extra_columns={"group": "group"}
    )(records(class_of(classes)))

    scores = one_step(
        TabularClassification(
            num_classes=classes,
            tabular_model=representation(),
            task_type="classification",
        ),
        batch,
        MultiClassMetrics(num_classes=classes),
    )

    assert 0.0 <= scores["mean_balanced_accuracy"] <= 1.0
    assert "calib_group_0_log_loss" in scores


def test_uplift_reaches_a_qini_number():
    batch = UpliftCollateFn(
        target_column="target", treatment_column="treatment", group_column="group"
    )(records(class_of(2)))

    model = SLearner(
        embedding=TabularEmbedding(
            num_numerical_features=N_NUM, hidden_size=WIDTH, vocab_size=VOCAB
        ),
        tabular_encoder=TabularTransformer(
            hidden_size=WIDTH, num_heads=2, num_layers=1
        ),
        # One token for treatment, one for the group.
        aggregation_config={
            "name": "linear",
            "num_features": N_CAT + N_NUM + 2,
            "emb_dim": WIDTH,
        },
        n_groups=N_GROUPS + 1,
        exchange_treatment_group=True,
    )

    scores = one_step(model, batch, UpliftMetrics())

    assert "mean_qini_auc_score" in scores
    assert "calib_group_0_qini_auc_score" in scores


def test_the_supervised_learner_scores_a_batch_with_no_targets():
    """Inference has no labels, and the signature has always said targets are optional."""
    batch = SupervisedCollateFn(target_column="target")(records(class_of(2)))
    model = SupervisedLearner(
        embedding=TabularEmbedding(
            num_numerical_features=N_NUM, hidden_size=WIDTH, vocab_size=VOCAB
        ),
        tabular_encoder=TabularTransformer(
            hidden_size=WIDTH, num_heads=2, num_layers=1
        ),
        aggregation_config={
            "name": "linear",
            "num_features": N_CAT + N_NUM,
            "emb_dim": WIDTH,
        },
    )
    model.eval()

    output = model(tab_features=batch["tab_features"], targets=None)

    assert output.logits.shape == (RECORDS, 1)
    assert output.loss is None


def test_the_supervised_learner_returns_logits_while_training():
    """Otherwise train_metrics have nothing to read."""
    batch = SupervisedCollateFn(target_column="target")(records(class_of(2)))
    model = SupervisedLearner(
        embedding=TabularEmbedding(
            num_numerical_features=N_NUM, hidden_size=WIDTH, vocab_size=VOCAB
        ),
        tabular_encoder=TabularTransformer(
            hidden_size=WIDTH, num_heads=2, num_layers=1
        ),
        aggregation_config={
            "name": "linear",
            "num_features": N_CAT + N_NUM,
            "emb_dim": WIDTH,
        },
    )
    model.train()

    output = model(**batch)

    assert output.logits.shape == (RECORDS, 1)
    assert output.loss.requires_grad
