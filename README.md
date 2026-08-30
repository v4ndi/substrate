# avatar

Фундаментальные модели для табличных и событийных данных о клиентах: табличный
трансформер с поддержкой внешних эмбеддингов, модели событийных
последовательностей, uplift- и multi-task-пайплайны и слой предобработки с двумя
взаимозаменяемыми бэкендами.

Обучение и инференс описываются одним YAML-конфигом и запускаются через
`torchrun`; всё остальное — сборка блоков внутри этого конфига.

## Установка

```bash
python -m pip install -e ".[spark,dev]"
```

| extra | что тянет | нужен для |
|---|---|---|
| `spark` | `pyspark==3.5.0` | `avatar.preprocessing.spark` (плюс JDK 8/11/17) |
| `catboost` | `catboost` | CatBoost-бенчмарк в `avatar.metrics.campaign` |
| `dev` | `pytest`, `ruff`, `pre-commit` | тесты и линтер |

`requirements.txt` — обёртка, ставящая `-e .[spark,catboost,dev]`;
авторитетный источник версий — `pyproject.toml`.

> Spark 3.5 не работает на JDK 21+. Укажите `JAVA_HOME` на JDK 8, 11 или 17.

## Пять минут

```bash
# предобработка на синтетических данных — работает сразу после установки
python examples/tabular_preprocessing/generate_data.py
python examples/tabular_preprocessing/run_both_backends.py

# обучение на своих данных
torchrun --standalone --nproc_per_node=8 -m avatar.train \
    --config-dir=configs --config-name=my_run

# инференс
python -m avatar.infer --config-dir=configs --config-name=inference
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
| `avatar/nn/` | строительные блоки моделей (табличные / событийные / эмбеддинги) |
| `avatar/pipeline/` | пайплайны задач (классификация, uplift, multi-task, next-k) |
| `avatar/preprocessing/` | бэкенды `spark` и `local` (pyarrow + numpy), общий `base` |
| `avatar/data/` | датасеты, collate-функции, чтение parquet |
| `avatar/metrics/` | метрики обучения, кампаний и uplift |
| `avatar/losses/` | функции потерь как отдельные подключаемые модули |
| `avatar/train/` | цикл обучения, колбэки, чекпоинты (Hydra + `torch.distributed`) |
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

## Бэкенды предобработки

`avatar.preprocessing` предоставляет один и тот же API на двух движках:

* `avatar.preprocessing.spark` — PySpark, для кластерного fit/transform;
* `avatar.preprocessing.local` — pyarrow + numpy, одна машина, потоковая
  обработка (переваривает данные больше объёма RAM), без Spark и JVM.

Артефакты (`dump()` / `load()`) переносятся между бэкендами в обе стороны. См.
`examples/tabular_preprocessing/` и `examples/eventsequence_preprocessing/`.
