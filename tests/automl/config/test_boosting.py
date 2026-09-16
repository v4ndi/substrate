import pytest

from avatar.automl.config.boosting import default_model_params, default_search_space


def test_default_model_params_match_lightautoml_boosters():
    catboost = default_model_params("catboost")
    xgboost = default_model_params("xgboost")

    assert catboost == {
        "thread_count": 4,
        "num_trees": 3000,
        "learning_rate": 0.03,
        "l2_leaf_reg": 1e-2,
        "bootstrap_type": "Bernoulli",
        "grow_policy": "SymmetricTree",
        "max_depth": 5,
        "min_data_in_leaf": 1,
        "one_hot_max_size": 10,
        "fold_permutation_block": 1,
        "boosting_type": "Plain",
        "boost_from_average": True,
        "od_type": "Iter",
        "od_wait": 100,
        "max_bin": 32,
        "feature_border_type": "GreedyLogSum",
        "nan_mode": "Min",
    }
    assert xgboost == {"n_estimators": 3000, "early_stopping_rounds": 100}


def test_default_search_spaces_have_expected_ranges():
    catboost = default_search_space(
        "catboost", n_trials=51, has_nan=True, has_categorical=True
    )
    xgboost = default_search_space("xgboost", n_trials=31)

    assert catboost == {
        "max_depth": {"type": "int", "low": 3, "high": 7, "step": 1},
        "nan_mode": {"type": "categorical", "choices": ["Max", "Min"]},
        "l2_leaf_reg": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "min_data_in_leaf": {"type": "int", "low": 1, "high": 20, "step": 1},
        "one_hot_max_size": {"type": "int", "low": 3, "high": 10, "step": 1},
    }
    assert xgboost == {
        "colsample_bytree": {
            "type": "categorical",
            "choices": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        "subsample": {"type": "categorical", "choices": [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]},
        "max_depth": {"type": "categorical", "choices": [5, 7, 9, 11, 13, 15, 17]},
        "learning_rate": {
            "type": "categorical",
            "choices": [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02],
        },
        "min_child_weight": {"type": "int", "low": 1, "high": 300, "step": 1},
        "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    }


def test_default_search_space_returns_an_isolated_copy():
    search_space = default_search_space("catboost", n_trials=3)
    search_space["max_depth"]["low"] = 999

    assert default_search_space("catboost", n_trials=3)["max_depth"]["low"] == 3


def test_default_model_params_return_an_isolated_copy():
    params = default_model_params("catboost")
    params["max_depth"] = 999

    assert default_model_params("catboost")["max_depth"] == 5


@pytest.mark.parametrize(
    ("train_rows", "learning_rate", "num_trees"),
    [
        (6_000, 0.02, 500),
        (6_001, 0.035, 5_000),
        (20_001, 0.03, 5_000),
        (50_001, 0.05, 2_000),
        (60_001, 0.045, 1_500),
        (100_001, 0.045, 3_000),
        (150_001, 0.045, 2_000),
        (300_001, 0.05, 3_000),
    ],
)
def test_catboost_binary_defaults_depend_on_train_rows(
    train_rows, learning_rate, num_trees
):
    params = default_model_params("catboost", task="binary", train_rows=train_rows)

    assert params["learning_rate"] == learning_rate
    assert params["num_trees"] == num_trees
    assert params["od_wait"] == 100


def test_catboost_multiclass_and_regression_defaults_depend_on_task_and_train_rows():
    small_multiclass = default_model_params(
        "catboost", task="multiclass", train_rows=100_000
    )
    large_multiclass = default_model_params(
        "catboost", task="multiclass", train_rows=100_001
    )
    regression = default_model_params("catboost", task="regression", train_rows=1)

    assert (small_multiclass["learning_rate"], small_multiclass["num_trees"]) == (
        0.03,
        3_000,
    )
    assert (large_multiclass["learning_rate"], large_multiclass["num_trees"]) == (
        0.03,
        4_000,
    )
    assert (
        regression["learning_rate"],
        regression["num_trees"],
        regression["od_wait"],
    ) == (0.05, 2_000, 300)


def test_xgboost_defaults_do_not_depend_on_task_or_train_rows():
    assert default_model_params(
        "xgboost", task="binary", train_rows=1
    ) == default_model_params("xgboost")
