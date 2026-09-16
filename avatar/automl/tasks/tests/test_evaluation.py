"""Shared identity-key alignment contract used by every supervised task."""

import polars as pl
import pytest

import avatar.automl.tasks.evaluation as evaluation_module
from avatar.automl import BinaryTaskConfig
from avatar.automl.data import CanonicalColumnMapper
from avatar.automl.exceptions import SchemaError
from avatar.automl.tasks.evaluation import align_prediction_scores, metric_slices


def _config(*, date_column="date", group_column="group") -> BinaryTaskConfig:
    return BinaryTaskConfig(
        env_type="local",
        backend="boosting",
        engine="catboost",
        device="cpu",
        target_column="target",
        client_id_column="epk_id",
        date_column=date_column,
        group_column=group_column,
        categorical_columns=(),
        numerical_columns=(),
        hidden_state_columns=(),
        model_layout="global" if group_column is not None else None,
        hyperopt=False,
        output_dir="outputs",
        environment={},
    )


@pytest.mark.parametrize(
    ("date_column", "group_column", "expected"),
    [
        (None, None, ()),
        (None, "group", (("group", ("group",)),)),
        ("date", None, (("date", ("date",)),)),
        ("date", "group", (("group", ("group",)), ("date_group", ("date", "group")))),
    ],
)
def test_metric_slice_matrix(date_column, group_column, expected):
    assert metric_slices(_config(date_column=date_column, group_column=group_column)) == expected


def test_alignment_without_date_uses_client_and_occurrence_only(caplog):
    config = _config(date_column=None, group_column=None)
    truth = pl.DataFrame({"epk_id": [1, 1], "target": [0, 1]})
    scores = pl.DataFrame({"epk_id": [1, 1], "score": [0.2, 0.8]})

    result = align_prediction_scores(
        scores,
        truth,
        config,
        CanonicalColumnMapper.from_config(config),
        score_columns=("score",),
    )

    assert result.select("target", "score").rows() == [(0, 0.2), (1, 0.8)]
    assert "duplicate rows by ['epk_id']" in caplog.text


@pytest.mark.parametrize(
    "score_columns",
    [
        ("score",),
        ("score_0", "score_1", "score_2"),
        ("score_s", "score_t", "score_x"),
    ],
)
def test_shared_alignment_rejects_equal_length_mismatched_keys(score_columns):
    config = _config()
    truth = pl.DataFrame(
        {
            "epk_id": [1, 2],
            "date": ["2026-01-01", "2026-01-01"],
            "group": ["a", "b"],
            "target": [0, 1],
        }
    ).with_columns(pl.col("date").str.to_date())
    values = {
        "epk_id": [1, 3],
        "date": ["2026-01-01", "2026-01-01"],
        **{column: [0.1, 0.9] for column in score_columns},
    }

    with pytest.raises(SchemaError, match="must match evaluation rows one-to-one"):
        align_prediction_scores(
            pl.DataFrame(values),
            truth,
            config,
            CanonicalColumnMapper.from_config(config),
            score_columns=score_columns,
        )


def test_shared_alignment_warns_for_duplicate_score_keys_before_rejecting_mismatch(caplog):
    config = _config()
    truth = pl.DataFrame(
        {
            "epk_id": [1, 2],
            "date": ["2026-01-01", "2026-01-01"],
            "group": ["a", "b"],
            "target": [0, 1],
        }
    ).with_columns(pl.col("date").str.to_date())
    scores = pl.DataFrame({"epk_id": [1, 1], "date": ["2026-01-01", "2026-01-01"], "score": [0.1, 0.9]})

    with pytest.raises(SchemaError, match="must match evaluation rows one-to-one"):
        align_prediction_scores(
            scores,
            truth,
            config,
            CanonicalColumnMapper.from_config(config),
            score_columns=("score",),
        )
    assert "Evaluation contains duplicate rows by ['epk_id', 'date']" in caplog.text
    assert len(caplog.records) == 1


def test_shared_alignment_rejects_legacy_model_scope_metadata():
    config = _config()
    truth = pl.DataFrame(
        {
            "epk_id": [1],
            "date": ["2026-01-01"],
            "group": ["a"],
            "target": [1],
        }
    ).with_columns(pl.col("date").str.to_date())
    scores = pl.DataFrame(
        {
            "epk_id": [1],
            "date": ["2026-01-01"],
            "model_scope": ["product"],
            "score": [0.8],
        }
    )

    with pytest.raises(SchemaError, match="incompatible legacy model_scope metadata"):
        align_prediction_scores(
            scores,
            truth,
            config,
            CanonicalColumnMapper.from_config(config),
            score_columns=("score",),
        )


def test_shared_alignment_warns_and_pairs_matching_duplicate_keys_by_occurrence(caplog):
    config = _config()
    truth = pl.DataFrame(
        {
            "epk_id": [1, 1, 2],
            "date": ["2026-01-01"] * 3,
            "group": ["a", "b", "a"],
            "target": [0, 1, 1],
        }
    ).with_columns(pl.col("date").str.to_date())
    scores = pl.DataFrame(
        {
            "epk_id": [1, 1, 2],
            "date": ["2026-01-01"] * 3,
            "score": [0.1, 0.8, 0.7],
        }
    )

    result = align_prediction_scores(
        scores,
        truth,
        config,
        CanonicalColumnMapper.from_config(config),
        score_columns=("score",),
    )

    assert result.select("target", "score").rows() == [(0, 0.1), (1, 0.8), (1, 0.7)]
    assert "Evaluation contains duplicate rows by ['epk_id', 'date']" in caplog.text
    assert len(caplog.records) == 1


def test_long_form_alignment_checks_keys_once_for_truth_and_once_for_all_scores(monkeypatch):
    config = _config()
    truth = pl.DataFrame(
        {
            "epk_id": [1, 2],
            "date": ["2026-01-01", "2026-01-01"],
            "group": ["a", "b"],
            "target": [0, 1],
        }
    ).with_columns(pl.col("date").str.to_date())
    scores = pl.DataFrame(
        {
            "epk_id": [1, 2, 1, 2],
            "date": ["2026-01-01"] * 4,
            "group": ["a", "b", "a", "b"],
            "model_layout": ["global", "global", "per_group", "per_group"],
            "score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    calls = []
    original = evaluation_module._check_global_keys

    def counted(*args, **kwargs):
        calls.append((kwargs["source"], kwargs.get("extra_keys", ())))
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation_module, "_check_global_keys", counted)

    result = align_prediction_scores(
        scores,
        truth,
        config,
        CanonicalColumnMapper.from_config(config),
        score_columns=("score",),
    )

    assert result.height == 4
    assert calls == [("evaluation data", ()), ("scores", ("model_layout",))]
