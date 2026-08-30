# Начало работы

Путь от чистого репозитория до обученной модели: установка → предобработка →
обучение → инференс. Первые два шага можно выполнить прямо сейчас на
синтетических данных; для третьего нужны свои данные.

## 1. Установка

```bash
git clone git@github.com:v4ndi/substrate.git avatar_fm
cd avatar_fm
python -m pip install -e ".[dev]"
```

Дополнительные наборы зависимостей:

| extra | что тянет | когда нужен |
|---|---|---|
| `spark` | `pyspark==3.5.0` | Spark-бэкенд предобработки; нужен JDK 8/11/17 |
| `catboost` | `catboost` | CatBoost-бенчмарк в `avatar.metrics.campaign` |
| `dev` | `pytest`, `ruff`, `pre-commit` | тесты и линтер |

Проверка, что всё встало:

```bash
python -m pytest -q
```

Spark-тесты сами пропускаются, если подходящей JVM нет — это нормально.

## 2. Предобработка

Модели не читают сырые таблицы. Препроцессор превращает произвольный parquet в
две упакованные колонки:

* `cat_features` — `list<int64>`, идентификаторы категорий со сквозной
  нумерацией, чтобы все категориальные колонки делили один `nn.Embedding`;
* `num_features` — `list<float32>`, стандартизованные числовые признаки.

Есть два взаимозаменяемых бэкенда с одинаковым API:
`avatar.preprocessing.local` (pyarrow + numpy, одна машина, потоковая обработка)
и `avatar.preprocessing.spark` (PySpark, кластер). Артефакты `dump()`/`load()`
переносятся между ними в обе стороны.

Запустить на синтетических данных прямо сейчас:

```bash
python examples/tabular_preprocessing/generate_data.py
python examples/tabular_preprocessing/run_both_backends.py
```

Скрипт обучает препроцессор обоими бэкендами (Spark пропускается, если нет JDK),
перекрёстно загружает артефакты и проверяет, что результаты совпадают.

На своих данных это выглядит так:

```python
import yaml
from avatar.preprocessing.local import TabularPreprocessor

pp = TabularPreprocessor(
    categorical_columns=cat_cols,
    numeric_columns=num_cols,
    spec_tokens={"pad": 0},
)
pp.fit("/data/train")                                    # один потоковый проход
pp.transform("/data/train", "/data/train_processed")
pp.transform("/data/valid", "/data/valid_processed")

yaml.safe_dump(pp.dump(), open("artifacts/preprocessor.yaml", "w"))
```

**Обучать препроцессор нужно только на train.** Для valid и test вызывается
`transform` тем же объектом — иначе статистики протекут между выборками.

Сохранённый артефакт понадобится на инференсе: его надо загрузить и применить
ровно тот же `transform`.

Для событийных последовательностей есть
`avatar.preprocessing.local.EventSequencePreprocessor` и пример
`examples/eventsequence_preprocessing/`.

## 3. Обучение

Обучение полностью описывается одним YAML. Минимальный рабочий конфиг:

```yaml
# configs/my_run.yaml
distributed:
  backend: null
  gradient_accumulation_steps: 1
amp: no

train_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: avatar.data.TabularDataset
    path: /data/train_processed
    shuffle_files: True
    shuffle_pq: True
  batch_size: 2048
  num_workers: 8
  drop_last: False
  pin_memory: True
  collate_fn:
    _target_: avatar.data.TabularCollateFn
    target_column: target

valid_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: avatar.data.TabularDataset
    path: /data/valid_processed
    shuffle_files: False
    shuffle_pq: False
  batch_size: 2048
  num_workers: 8
  drop_last: False
  pin_memory: True
  collate_fn:
    _target_: avatar.data.TabularCollateFn
    target_column: target

model:
  _target_: avatar.pipeline.tabular.TabularClassification
  tabular_model:
    _target_: avatar.pipeline.tabular.TabularWithAggregatedStates
    embedding:
      _target_: avatar.nn.embedding.TabularEmbedding
      num_numerical_features: 24    # сколько числовых признаков после препроцессинга
      vocab_size: 74                # pp.vocab_size
      hidden_size: 64
    encoder:
      _target_: avatar.nn.tabular.TabularTransformer
      hidden_size: 64
      num_heads: 4
      num_layers: 3
    aggregation_config:
      name: linear
      num_features: 32              # категориальных + числовых признаков
      emb_dim: 64
  num_classes: 2
  task_type: classification

optimizer:
  _target_: torch.optim.AdamW
  _partial_: True
  lr: 0.001

scheduler:
  _target_: transformers.optimization.get_scheduler
  _partial_: True
  name: cosine
  num_warmup_steps: 1000
  num_training_steps: 100000

train:
  num_epochs: 20
  seed: 42
  max_saved_checkpoints: 5
  early_stopping:
    _target_: avatar.train.EarlyStopping
    main_metric: roc_auc_score
    patience: 5
    strategy: max

mlflow:
  experiment_name: my_experiment
  run_name: debug

metrics:
  valid_metrics:
    _target_: avatar.metrics.RocAucScore
```

Три числа, которые чаще всего ставят неправильно, берутся из препроцессора:
`vocab_size` — это `pp.vocab_size`, `num_numerical_features` — длина
`num_features`, а `num_features` в `aggregation_config` — общее число токенов,
приходящих на агрегацию, то есть категориальные плюс числовые (плюс единица за
каждый внешний эмбеддинг, подмешанный как признак).

Запуск:

```bash
# одна карта
python -m avatar.train --config-dir=configs --config-name=my_run

# все карты машины
torchrun --standalone --nproc_per_node=8 -m avatar.train \
    --config-dir=configs --config-name=my_run
```

Любой ключ переопределяется из командной строки — удобно для быстрой проверки:

```bash
python -m avatar.train --config-dir=configs --config-name=my_run train.num_epochs=1
```

### Куда всё складывается

* Чекпоинты — в `best_models/{experiment_name}/{run_name}/{шаг}/`.
* Метрики — в MLflow, по умолчанию в `./mlruns` рядом с местом запуска.
* Если `run_name` не `debug`, а каталог чекпоинтов уже существует, запуск
  упадёт на старте. Это защита от затирания чужого прогона.

## 4. Инференс

```yaml
# configs/inference.yaml
load_state: best_models/my_experiment/debug/42/model.bin
distributed:
  backend: null
amp: no
test_dataloader:
  ...                    # как valid_dataloader, но по тестовой выборке
model:
  ...                    # ровно тот же блок, что при обучении
metrics:
  test_metrics:
    _target_: avatar.metrics.ClassificationInferenceMetrics
    input_columns_to_save: [epk_id, targets]
    path_to_save: predicts/my_run
```

```bash
python -m avatar.infer --config-dir=configs --config-name=inference
```

Блок `model:` должен совпадать с обучающим до последнего аргумента — иначе
загрузка весов не сойдётся по именам параметров.

## Что дальше

* [configuration/schema.md](configuration/schema.md) — все ключи конфига.
* [configuration/targets.md](configuration/targets.md) — что можно ставить в
  `_target_`.
* [configuration/distributed.md](configuration/distributed.md) — multi-GPU,
  несколько узлов, смешанная точность.
* [../examples/README.md](../examples/README.md) — сквозные примеры.
