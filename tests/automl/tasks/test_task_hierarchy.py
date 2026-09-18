import inspect

import pytest

from fmlib.automl import (
    BinaryTask,
    BinaryTaskConfig,
    MulticlassTask,
    MulticlassTaskConfig,
    RegressionTask,
    RegressionTaskConfig,
    ResponseTask,
    ResponseTaskConfig,
    UpliftTask,
    UpliftTaskConfig,
)
from fmlib.automl.exceptions import ConfigError, UnsupportedBackendError
from fmlib.automl.tasks.base import BaseTask
from fmlib.automl.tasks.calibration import CalibratableTask
from fmlib.automl.tasks.supervised import SupervisedTask


def _config(config_class, tmp_path):
    values = {
        "env_type": "local",
        "backend": "boosting",
        "engine": "catboost",
        "device": "cpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ("feature",),
        "hidden_state_columns": (),
        "model_layout": "global",
        "hyperopt": False,
        "output_dir": tmp_path,
        "environment": {},
    }
    if config_class in {ResponseTaskConfig, UpliftTaskConfig}:
        values.update(treatment_column="treatment", inverse_treatment=True)
    if config_class is UpliftTaskConfig:
        values["estimate_propensity"] = False
    return config_class(**values)


def test_task_hierarchy_keeps_response_as_the_binary_specialization():
    assert BinaryTask.__bases__ == (SupervisedTask,)
    assert ResponseTask.__bases__ == (CalibratableTask, BinaryTask)
    assert RegressionTask.__bases__ == (SupervisedTask,)
    assert MulticlassTask.__bases__ == (SupervisedTask,)
    assert UpliftTask.__bases__ == (CalibratableTask, BaseTask)
    assert inspect.isabstract(BaseTask)
    assert issubclass(ResponseTask, BinaryTask)
    assert not issubclass(RegressionTask, BinaryTask)
    assert not issubclass(BinaryTask, RegressionTask)
    assert not issubclass(MulticlassTask, BinaryTask | RegressionTask)
    assert not issubclass(UpliftTask, BinaryTask | RegressionTask | MulticlassTask)


def test_common_lifecycle_does_not_require_supervised_metrics():
    assert not issubclass(UpliftTask, SupervisedTask)
    for name in ("_metric", "_optimization_metric"):
        assert not hasattr(BaseTask, name)
        assert not hasattr(UpliftTask, name)
        assert hasattr(SupervisedTask, name)
    assert not inspect.isabstract(UpliftTask)


def test_common_task_base_has_no_task_specific_treatment_contract():
    assert "_apply_treatment_convention" not in BaseTask.__dict__


@pytest.mark.parametrize(
    ("task_class", "config_class"),
    [
        (task_class, config_class)
        for task_class, expected_config in (
            (BinaryTask, BinaryTaskConfig),
            (RegressionTask, RegressionTaskConfig),
            (MulticlassTask, MulticlassTaskConfig),
            (ResponseTask, ResponseTaskConfig),
            (UpliftTask, UpliftTaskConfig),
        )
        for config_class in (
            BinaryTaskConfig,
            RegressionTaskConfig,
            MulticlassTaskConfig,
            ResponseTaskConfig,
            UpliftTaskConfig,
        )
        if config_class is not expected_config
    ],
)
def test_task_rejects_another_tasks_config(task_class, config_class, tmp_path):
    config = _config(config_class, tmp_path)

    with pytest.raises(
        ConfigError,
        match=rf"{task_class.__name__} requires .*Config; got {config_class.__name__}",
    ):
        task_class(config)


@pytest.mark.parametrize(
    ("task_class", "config_class"),
    [
        (BinaryTask, BinaryTaskConfig),
        (RegressionTask, RegressionTaskConfig),
        (MulticlassTask, MulticlassTaskConfig),
        (ResponseTask, ResponseTaskConfig),
        (UpliftTask, UpliftTaskConfig),
    ],
)
def test_task_accepts_its_own_config(task_class, config_class, tmp_path):
    config = _config(config_class, tmp_path)

    assert isinstance(task_class(config).config, config_class)


def test_evaluate_api_reads_target_only_from_the_dataset():
    for task_class in (
        BinaryTask,
        RegressionTask,
        MulticlassTask,
        ResponseTask,
        UpliftTask,
    ):
        parameters = inspect.signature(task_class.evaluate).parameters
        assert list(parameters) == [
            "self",
            "test_path",
            "scores",
            "metrics",
            "env_type",
            "device",
            "environment",
        ]
        assert parameters["env_type"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["env_type"].default is None
        assert parameters["device"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["device"].default is None
        assert parameters["environment"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["environment"].default is None


def test_persisted_result_loaders_use_parallel_public_names():
    assert hasattr(BaseTask, "load_prediction")
    assert hasattr(BaseTask, "load_evaluation")
    assert not hasattr(BaseTask, "evaluation")


def test_every_task_resolves_its_backend_family_through_one_dictionary():
    """The dispatch point P6 asks for: a family name in, an adapter class out."""
    for task_class in (
        BinaryTask,
        MulticlassTask,
        RegressionTask,
        ResponseTask,
        UpliftTask,
    ):
        assert "boosting" in task_class._backend_loaders
        adapter = task_class._backend_class_for("boosting")
        assert adapter is task_class._backend_loaders["boosting"]()


@pytest.mark.parametrize(
    "task_class", [BinaryTask, MulticlassTask, RegressionTask, ResponseTask]
)
def test_the_supervised_tasks_can_reach_the_tabnn_adapter(task_class):
    adapter = task_class._backend_class_for("tabnn")
    assert adapter.__name__ == "TabNNBackend"


def test_a_task_without_a_tabnn_adapter_says_what_it_supports():
    """Uplift arrives later; until then the refusal names the task and the list."""
    with pytest.raises(UnsupportedBackendError) as error:
        UpliftTask._backend_class_for("tabnn")
    message = str(error.value)
    assert "'uplift'" in message
    assert "'tabnn'" in message
    assert "boosting" in message
