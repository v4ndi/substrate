# avatar AutoML: запуск и справочник API

Рабочие notebooks: `binary_pipeline.ipynb`, `response_pipeline.ipynb`, `regression_pipeline.ipynb`, `multiclass_pipeline.ipynb`, `uplift_pipeline.ipynb`. Каждый показывает typed config, train, одну ячейку вариантов `model_layout`, predict/evaluate, persistence и Osiris. Только Response/Uplift содержат одну отдельную path-based calibration ячейку.

## Запуск

### Установка

AutoML устанавливается вместе с `avatar`; нужны Python 3.11+ и extra `boosting`
(CatBoost и XGBoost подключаются лениво, без них движки поднимают
`MissingDependencyError`):

```bash
python -m venv .venv
.venv/bin/pip install -e ".[boosting,dev]"
```

Чтобы notebook получил kernel, один раз установите `ipykernel` в это окружение:

```bash
.venv/bin/python -m pip install "ipykernel>=6.29,<8"
.venv/bin/python -m ipykernel install \
  --user \
  --name avatar-env \
  --display-name "avatar automl env"
```

Для Osiris (`env_type="osiris"`) `environment.venv_path` должен указывать на
shared окружение на кластере, а не на локальный `.venv`.

Чтобы до обучения импортировать Osiris, например для просмотра логов, выполните:

```bash
printf '%s\n' /opt/clients > \
  "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/osiris_clients.pth" \
  && python -c 'import osiris; print(osiris.__file__)'
```

### Local

```python
from pathlib import Path
from avatar.automl import BinaryTask, BinaryTaskConfig

config = BinaryTaskConfig(
    env_type="local",
    backend="boosting",
    engine="catboost",
    device="gpu",
    target_column="target_attr_1",
    client_id_column="epk_id",
    group_column="target_attr_2",
    date_column="report_month",
    categorical_columns=["category"],
    numerical_columns=["amount"],
    hidden_state_columns=["seq_hidden_state"],
    model_layout="global",
    hyperopt=False,
    optimization_metric="roc_auc",
    output_dir=Path("outputs/binary"),
)

task = BinaryTask(config)
training = task.train("data/train", "data/valid")
prediction = task.predict("data/test")
evaluation = task.evaluate(
    "data/test",
    prediction,
    metrics=["roc_auc", "precision@10", "recall@10"],
)
artifact_path = task.save("models/binary_final")
```

Local operation хранит owner и heartbeat. Если kernel/process завершился во время operation, после `TaskClass.load(<output_dir>/automl/<id>)` вызов `status()` помечает её `interrupted`; operation можно запустить повторно.

### Osiris

Data, `output_dir` и artifacts должны быть доступны notebook и job. Default launcher использует shared `/home/datalab/nfs/sber-amazme-fmlib/env`.

```python
from avatar.automl import BinaryTask, BinaryTaskConfig, EnvironmentConfig

config = BinaryTaskConfig(
    env_type="osiris",
    backend="boosting",
    engine="catboost",
    device="gpu",
    target_column="target_attr_1",
    client_id_column="epk_id",
    group_column="target_attr_2",
    date_column="report_month",
    categorical_columns=[],
    numerical_columns=["amount"],
    hidden_state_columns=["seq_hidden_state"],
    model_layout="global",
    hyperopt=False,
    output_dir=Path("outputs/binary_remote"),
    environment=EnvironmentConfig(pool="common"),
)

task = BinaryTask(config)
task.train("/shared/data/train", "/shared/data/valid")
task.status(wait=True)
prediction = task.predict("/shared/data/test", env_type="osiris")
task.status(wait=True)
prediction = task.load_prediction("/shared/data/test")
```

`pool=None` или `"common"` даёт standard profile 1 node/1 GPU. Custom pool требует explicit `num_nodes` и `num_gpus`, например `EnvironmentConfig(pool="b2c", num_nodes=1, num_gpus=4)`.

## Конфигурация

| Поле | Значения/default и назначение |
|---|---|
| `env_type` | Явно `"local"` или `"osiris"`. |
| `backend` | Обязательное `"boosting"`; TabNN/STE зарезервирован. |
| `engine` | `"catboost"` или `"xgboost"`. |
| `device` | Explicit CPU/GPU; Osiris только GPU. |
| `target_column` | Train/evaluate target; production predict может не содержать target. |
| `client_id_column` | Обязательная identity column. |
| `group_column` | Имя или `None`; настроенная role обязательна в релевантных datasets. |
| `date_column` | Имя или `None`; поддержанные string/date formats нормализуются, date не входит в features. |
| `treatment_column`, `inverse_treatment` | Только Response/Uplift. Response treatment optional; Uplift treatment и bool inversion обязательны. |
| `categorical_columns` | Sequence либо absolute YAML path; пусто допустимо, null допустим. |
| `numerical_columns` | Sequence либо absolute YAML path; null допустим, infinity запрещён. |
| `hidden_state_columns` | Fixed-size List/Array Float32/Float64, разворачивается в Float32 features. |
| `model_layout` | При group: `global`, `per_group`, `global_and_per_group`; без group поле запрещено. Layout с per-group branch отклоняет unknown inference group. |
| `hyperopt` | False — `model_params`; True — Optuna. |
| `optimization_metric` | Зарегистрированное имя метрики выбора модели; `None` разрешается в task default. Callable не поддерживается. |
| `verbose` | bool или integer ≥ 0, default True. |
| `model_params` | Только без hyperopt; task-owned runtime/objective/logging params запрещены. |
| `search_space` | Только с hyperopt; explicit mapping целиком заменяет dynamic default. |
| `n_trials` | Только с hyperopt; default 50. |
| `random_state` | Integer, default 42. |
| `output_dir` | Root entities/artifacts/results/reports/logs. |
| `environment` | Optional `EnvironmentConfig`; operation-only override config/artifact не мутирует. |
| `estimate_propensity` | Explicit bool только Uplift. |

### Optimization metric и objective Optuna

| Task | Доступные `optimization_metric` | Default/direction |
|---|---|---|
| Binary | `roc_auc` | `roc_auc`, maximize |
| Response | `roc_auc` | `roc_auc`, maximize |
| Regression | `mse`, `mae` | `mse`, minimize |
| Multiclass | `roc_auc_ovr_macro`, `accuracy`, `f1_macro`, `log_loss` | `roc_auc_ovr_macro`; первые три maximize, `log_loss` minimize |
| Uplift | `qini_auc`, `uplift_auc` | `qini_auc`, maximize |

Реестр метрик является единым источником допустимых имён и направлений. `optimization_metric` используется только для validation и выбора trial; native loss/eval metric backend-а остаётся внутренней task-owned настройкой. Произвольные notebook-callable не принимаются ни local, ни Osiris.

Default Optuna space динамичен: CatBoost учитывает trial budget, NaN и categorical schema фактической model part; XGBoost добавляет regularization dimensions только при `n_trials > 30`. Explicit search space имеет приоритет.

### environment

| Поле | Значения/default и назначение |
|---|---|
| `image` | Container image, default `registry.ca.sbrf.ru/ci02684173/ci02697916/notebooks/python3.12/cuda12.4/d-03.000.00:d-03.000.00-gigachat`. |
| `pool` | `None`/`"common"` — standard; другая непустая строка — custom. |
| `num_gpus`, `num_nodes` | Standard фиксирован 1/1; custom требует оба positive integer. |
| `venv_path` | Absolute shared fmlib venv; default `/home/datalab/nfs/sber-amazme-fmlib/env`. |
| `env` | String mapping; `PYTHONPATH`/`CUDA_VISIBLE_DEVICES` launcher-controlled. |
| `poll_interval_seconds` | Positive number, default 30.0. |
| `log_dir` | Path/None; default `<output_dir>/logs`. |

## Публичный lifecycle

Общий для пяти matching typed tasks.

| API | Контракт |
|---|---|
| `TaskClass(config)` | Создаёт persistent entity с ID/path. |
| `train(train_path, valid_path, ...)` | Local → `TrainingResult`; Osiris → None, затем wait/result load. |
| `training_result()` | Загружает training summary. |
| `predict(test_path, ..., model_layout=None)` | Только raw `PredictionResult`; Osiris result через `load_prediction(test_path)`. |
| `load_prediction(test_path)` | Exact canonical path lookup. |
| `calibrate(test_path, calibration_path, calibration_strategy=..., env_type=None, environment=None)` | Только Response/Uplift; пути должны различаться, для обоих требуется ранее завершённый `predict`; загружает два сохранённых prediction и сама prediction не запускает; local → `CalibrationResult`, Osiris → None. |
| `load_calibration(test_path)` | Exact canonical test-path lookup. |
| `evaluate(test_path, scores=None, *, metrics=None, ...)` | Один вызов оценивает один вид scores: `PredictionResult` → `*_raw`, допустимый `CalibrationResult` → `*_calibrated`; `metrics=None` означает фиксированный task default, явный непустой список полностью задаёт пользовательские метрики; разрешён зарегистрированный task parquet path, `scores=None` означает stored prediction. |
| `load_evaluation(test_path)` | Exact persisted evaluation; после двух вызовов для одного `test_path` содержит обе независимые половины. |
| `status(wait=False)` | Operation/scheduler states; confirmed dead local owner → interrupted. |
| `save(path=None, overwrite=False)` | Standalone snapshot; train уже сохранил entity artifact. Существующий destination заменяется только при явном `overwrite=True`. |
| `TaskClass.load(path)` | Entity сохраняет ID/lifecycle; standalone artifact создаёт independent entity. |

После смерти kernel выбирайте entity path для продолжения прежнего lifecycle либо standalone artifact для новой independent entity.

## Task-specific контракты

| Task | Target/train | Prediction | Evaluation | Calibration |
|---|---|---|---|---|
| Binary | `0/1`, treatment fields запрещены. | Raw `score`. | ROC AUC; precision/recall top-k; role slices. | Нет. |
| Response | `0/1`; treatment optional categorical feature. | Raw `score`. | Binary metrics. | Beta/isotonic, один mapping на branch. |
| Regression | Finite numerical; treatment fields запрещены. | Continuous `score`. | MSE, MAE, MAPE; role slices. | Нет. |
| Multiclass | Минимум 3 scalar labels; treatment fields запрещены. | `score_0...score_K-1`, class order. | `roc_auc_ovr_macro`, log loss, accuracy, F1; `class_wise_roc_auc` и class-wise top-k. | Нет. |
| Uplift | Outcome/treatment `0/1`, обе arms/classes. | S/T/X effects, potential outcomes, propensity. | Qini/AUUC, uplift@k, arm ROC AUC. | Beta/isotonic mappings каждой S/T/X arm внутри branch. |

Metric slices: без date/group только overall; group добавляет group; date добавляет date; обе роли дают group и date × group. Combined layout дополнительно разделяет каждый применимый slice по branch.

Стандартный `metrics=None` сохраняет полный набор из таблицы задачи; для Binary/Response top-k содержит `k = 5, 10, 20, 25, 50`, для Multiclass те же class-wise top-k и `class_wise_roc_auc`, для Uplift — `uplift_at_10/20/50`, `treatment_roc_auc` и `control_roc_auc` для S/T/X. Явный список применяется одинаково к overall и всем применимым slices, а также к raw и calibrated scores. Пустой список, повторяющиеся/неизвестные/несовместимые имена и строка вместо sequence отклоняются до запуска operation. Диагностические `n_samples`, `n_positives`, `n_treatment` и `n_control` не являются выбираемыми метриками.

`EvaluationResult` полностью разделён по виду scores: `metrics_raw`/`metrics_calibrated`, `metrics_by_group_raw`/`metrics_by_group_calibrated`, `metrics_by_class_raw`/`metrics_by_class_calibrated`, `figures_raw`/`figures_calibrated`, `excel_paths_raw`/`excel_paths_calibrated`. Несуффиксированных полей нет. Чтобы получить оба набора, вызовите `evaluate(test_path, prediction)` и `evaluate(test_path, calibration)` с одним `test_path`; результат и `<artifact>/evaluation_result.json` накопительно содержат оба набора.

## `output_dir` и artifacts

При создании task возникает `<output_dir>/automl/<automl_id>`; один root хранит несколько независимых entities.

```text
<output_dir>/
├── automl/<automl_id>/
│   ├── entity.json; datasets.json; training_result.json                 # после train
│   ├── artifact/
│   │   ├── config.json; manifest.json
│   │   └── model/<model_index>/
│   │       ├── backend.json
│   │       ├── model.cbm | model.json                                  # обычная задача
│   │       └── components/<component>/...                              # uplift S/T/X/propensity
│   ├── operations/{train|predict|calibrate|evaluate}/<run_id>/operation.json
│   ├── predictions/<dataset_key>/scores.parquet; prediction.json
│   ├── calibrations/<test_dataset_key>/scores.parquet; calibration.json
│   ├── evaluations/<dataset_key>/
│   │   ├── evaluation_result.json                                     # raw + calibrated metrics/tables
│   │   ├── metrics_raw.xlsx; metrics_calibrated.xlsx
│   │   ├── metrics_by_class_raw.xlsx; metrics_by_class_calibrated.xlsx # multiclass
│   │   ├── feature_importance_raw.xlsx; feature_importance_calibrated.xlsx
│   │   └── <task-specific figure>_raw.png; <task-specific figure>_calibrated.png
│   ├── .remote_parts/<run_id>/part-<index>/...                          # remote train parts
│   └── operations/<action>/<run_id>/job-<index>/                        # remote only
│       ├── run_spec.json; submit.log; remote.log; result.json
│       └── scores.parquet                                               # worker predict
├── artifacts/{binary|response|multiclass|regression|uplift}_model/               # default save()
└── logs/<run_id>.<train|predict|calibrate|evaluate>.log                            # local
```

`dataset_key` = имя + hash canonical absolute input path; mapping хранится в `<artifact>/datasets.json`. Prediction/calibration/evaluation JSON содержат typed metadata и ссылки на свои parquet/Excel/PNG results. Grouped metric tables находятся внутри `<artifact>/evaluation_result.json`; отдельных parquet для них нет, а Excel `metrics_raw.xlsx`/`metrics_calibrated.xlsx` включает overall и grouped slices.

Artifact: `<artifact>/config.json` — typed config; `<artifact>/manifest.json` — task/backend/engine, `optimization_metric`, feature schemas/dimensions, model layout/group каждой модели, best params, validation metrics, source manifests, task-specific state; `<artifact>/backend.json` — estimator params/category dictionaries; `<artifact>/model.cbm`/`<artifact>/model.json` — native CatBoost/XGBoost binary. `load()` проверяет согласованность manifest/config/models.

Figures используют role-first names: binary/response ROC AUC, multiclass metrics/confusion matrices, regression metrics, uplift Qini/uplift curves; date figures имеют suffix `_by_date`, branch figures — global/per_group.

`save(custom_path)` пишет тот же standalone artifact; default tree создаётся только `save()` без path. Существующий путь по умолчанию не перезаписывается: для нового snapshot укажите другой путь либо явно передайте `overwrite=True`. При заданном `environment.log_dir` local logs идут туда, не в `<output_dir>/logs`.

## Добавление метрик и калибровок

Новая метрика должна быть частью общей editable installation `avatar`, доступной и notebook, и Osiris worker; notebook-callable не поддерживаются. Реализацию добавляют в task-specific модуль `avatar/automl/metrics/` по готовым классам из `avatar/automl/metrics/binary.py`, `avatar/automl/metrics/regression.py`, `avatar/automl/metrics/multiclass.py` или `avatar/automl/metrics/uplift.py`. Класс структурно соблюдает `Metric` и принимает `MetricInput` из `avatar/automl/metrics/base.py`, задаёт стабильные `name`, `supported_tasks`, `optimization_direction` (`None` для evaluation-only) и реализует `compute()`. Готовый экземпляр регистрируют в единственном реестре `avatar/automl/metrics/__init__.py`. Математику и входной scalar/matrix/treatment-aware контракт проверяют в `tests/automl/metrics/test_metrics.py`, а config, training/evaluation, persistence и remote run spec — в соответствующих task-level tests.

Новую стратегию калибровки создают в `avatar/automl/calibrators/<strategy>.py` по `avatar/automl/calibrators/beta_calibration.py` и `avatar/automl/calibrators/isotonic_regression.py`. Класс структурно соблюдает `Calibrator` из `avatar/automl/calibrators/base.py`: реализует `fit`, `predict`, `to_dict` и `from_dict`, а сохраняемый state остаётся неизменяемым и JSON-safe. Публичное имя добавляют в `CalibrationStrategy` и `calibrator_class()` в `avatar/automl/calibrators/__init__.py`, затем покрывают Response/Uplift, persistence и fake-Osiris tests. `avatar/automl/tasks/calibration.py` не требует изменения только тогда, когда стратегия соблюдает уже существующий scalar mapping contract.
