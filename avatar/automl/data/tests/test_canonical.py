import polars as pl
import pytest

from avatar.automl import ResponseTaskConfig
from avatar.automl.data import CanonicalColumnMapper
from avatar.automl.exceptions import SchemaError


def _config():
    return ResponseTaskConfig(
        env_type="local",
        backend="boosting",
        engine="catboost",
        device="cpu",
        target_column="label",
        client_id_column="client",
        group_column="channel_cd",
        treatment_column="cg_flg",
        inverse_treatment=True,
        date_column="period",
        categorical_columns=["channel_cd", "cg_flg"],
        numerical_columns=["amount"],
        hidden_state_columns=(),
        model_layout="global",
        hyperopt=False,
        output_dir="outputs",
        environment={},
    )


def test_canonical_mapper_normalizes_config_frame_and_restores_result_names():
    mapper = CanonicalColumnMapper.from_config(_config())
    internal = mapper.normalize_config(_config())
    frame = mapper.normalize_frame(
        pl.DataFrame(
            {
                "label": [0],
                "client": [1],
                "channel_cd": ["a"],
                "cg_flg": [1],
                "period": ["2026-01-01"],
                "amount": [1.0],
            }
        )
    )

    assert internal.target_column == "target"
    assert internal.categorical_columns == ("group", "treatment")
    assert {"target", "epk_id", "group", "treatment", "date"} <= set(frame.columns)
    assert mapper.restore_frame(frame.select("epk_id", "date")).columns == ["client", "period"]


def test_canonical_mapper_rejects_an_unrelated_canonical_name_collision():
    mapper = CanonicalColumnMapper.from_config(_config())
    frame = pl.DataFrame({"label": [0], "target": [42]})

    with pytest.raises(SchemaError, match="collide"):
        mapper.normalize_frame(frame)
