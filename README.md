# fmlib

Фундаментальные модели для табличных и событийных данных о клиентах: табличный
трансформер с поддержкой внешних эмбеддингов, модели событийных
последовательностей, uplift- и multi-task-пайплайны и слой предобработки с двумя
взаимозаменяемыми бэкендами, плюс `fmlib.automl` — автоматическое обучение
типовых supervised-задач.

Обучение и инференс описываются одним YAML-конфигом и запускаются через
`torchrun`; всё остальное — сборка блоков внутри этого конфига.

## Установка

```bash
python -m pip install -e ".[spark,dev]"
```

| extra | что тянет | нужен для |
|---|---|---|
| `spark` | `pyspark==3.5.0` | `fmlib.preprocessing.spark` (плюс JDK 8/11/17) |
| `catboost` | `catboost` | CatBoost-бенчмарк в `fmlib.metrics.campaign` |
| `boosting` | `catboost`, `xgboost` | движки бустинга в `fmlib.automl` |
| `dev` | `pytest`, `ruff`, `pre-commit` | тесты и линтер |

`requirements.txt` — обёртка, ставящая `-e .[spark,boosting,dev]`;
авторитетный источник версий — `pyproject.toml`.

> Spark 3.5 не работает на JDK 21+. Укажите `JAVA_HOME` на JDK 8, 11 или 17.

## Пять минут

```bash
# предобработка на синтетических данных — работает сразу после установки
python examples/tabular_preprocessing/generate_data.py
python examples/tabular_preprocessing/run_both_backends.py

# обучение на своих данных
torchrun --standalone --nproc_per_node=8 -m fmlib.train \
    --config-dir=configs --config-name=my_run

# инференс
python -m fmlib.inference --config-dir=configs --config-name=inference
```

Полный путь с разбором конфига — в
[docs/getting_started.md](docs/getting_started.md).

## Документация

| раздел | о чём |
|---|---|
| [docs/getting_started.md](docs/getting_started.md) | установка → предобработка → обучение → инференс |
| [docs/configuration/schema.md](docs/configuration/schema.md) | все ключи конфига, значения по умолчанию, ловушки |
| [docs/configuration/targets.md](docs/configuration/targets.md) | каталог классов для `_target_` |
| [docs/configuration/distributed.md](docs/configuration/distributed.md) | multi-GPU и несколько узлов, AMP, DDP |
| [docs/README.md](docs/README.md) | навигация по всей документации |
| [examples/README.md](examples/README.md) | сквозные примеры и что им нужно из данных |

## Раскладка

| путь | что |
|---|---|
| `fmlib/nn/` | строительные блоки моделей (табличные / событийные / эмбеддинги) |
| `fmlib/pipeline/` | пайплайны задач (классификация, uplift, multi-task, next-k) |
| `fmlib/preprocessing/` | бэкенды `spark` и `local` (pyarrow + numpy), общий `base` |
| `fmlib/data/` | датасеты, collate-функции, чтение parquet |
| `fmlib/metrics/` | метрики обучения, кампаний и uplift |
| `fmlib/losses/` | функции потерь как отдельные подключаемые модули |
| `fmlib/train/` | цикл обучения, колбэки, чекпоинты (Hydra + `torch.distributed`) |
| `fmlib/automl/` | автоматическое обучение типовых задач (см. ниже) |
| `examples/` | сквозные примеры |
| `docs/` | документация |

## Тесты

```bash
python -m pytest                 # по умолчанию: без @slow; Spark-тесты пропускаются без JDK
python -m pytest -m slow         # длинные многопроцессные прогоны шардирования
python -m pytest tests/docs      # проверки документации: ссылки, таргеты, пути
JAVA_HOME=/path/to/jdk17 python -m pytest tests/spark tests/local
```

`tests/conftest.py` сам ищет подходящую для Spark JVM (проверяет `$SPARK_JDK`,
типовые пути sdkman и системы, затем `$JAVA_HOME`); если ничего не нашлось,
Spark-тесты пропускаются, а не падают.

## AutoML

`fmlib.automl` обучает, скорит, оценивает и сохраняет модель для типовой
supervised-задачи по путям к parquet и одному типизированному конфигу —
цикл обучения писать не нужно. Задачи: `BinaryTask`, `ResponseTask`,
`RegressionTask`, `MulticlassTask`, `UpliftTask`.

```python
from fmlib.automl import BinaryTask, BinaryTaskConfig

task = BinaryTask(BinaryTaskConfig(
    env_type="local", backend="boosting", engine="catboost", device="cpu",
    target_column="target", client_id_column="epk_id",
    group_column=None, date_column=None,
    categorical_columns=["cat_1"], numerical_columns=["num_1"],
    hidden_state_columns=[],
    hyperopt=True, n_trials=20, output_dir="outputs/binary",
))
task.train(train_path, valid_path)
scores = task.predict(test_path)
report = task.evaluate(test_path)
```

Модуль работает на polars и не зависит от torch-части библиотеки: глобальная
модель, модели по группам или и то и другое (`model_layout`), поиск
гиперпараметров Optuna, изотоническая и бета-калибровка, графики и
Excel-отчёты, локальный запуск или Osiris. Сейчас единственный бэкенд —
`boosting` (CatBoost / XGBoost); `backend="tabnn"` зарезервирован под
нейросетевую реализацию.

Пять ноутбуков-пайплайнов и справочник по конфигам, метрикам и калибраторам —
в `examples/automl/`.

## Бэкенды предобработки

`fmlib.preprocessing` предоставляет один и тот же API на двух движках:

* `fmlib.preprocessing.spark` — PySpark, для кластерного fit/transform;
* `fmlib.preprocessing.local` — pyarrow + numpy, одна машина, потоковая
  обработка (переваривает данные больше объёма RAM), без Spark и JVM.

Артефакты (`dump()` / `load()`) переносятся между бэкендами в обе стороны. См.
`examples/tabular_preprocessing/` и `examples/eventsequence_preprocessing/`.
