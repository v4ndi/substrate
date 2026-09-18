# Схема конфига запуска

Обучение и инференс полностью описываются одним YAML-файлом. Здесь перечислены
все ключи, которые читает код: что означают, что будет по умолчанию и что
произойдёт, если ключ не указать.

Каталог классов, которые можно подставлять в `_target_`, — в
[targets.md](targets.md). Блоки `distributed:`, `amp:`, `ddp:` и `compile:`
разобраны отдельно в [distributed.md](distributed.md).

## Как конфиг попадает в запуск

```bash
torchrun --standalone --nproc_per_node=2 -m fmlib.train \
    --config-dir=configs --config-name=my_run
```

Конфигурация читается через Hydra: `--config-dir` — папка с YAML,
`--config-name` — имя файла без расширения. Любой ключ можно переопределить из
командной строки:

```bash
python -m fmlib.train --config-dir=configs --config-name=my_run \
    train.num_epochs=1 optimizer.lr=0.0005
```

Везде, где в схеме встречается `_target_`, работает механизм Hydra
`instantiate`: значение — полное имя класса, остальные ключи блока передаются в
его конструктор. `_partial_: True` означает «не создавать объект, а вернуть
функцию, которую вызовут позже» — так задаются оптимизатор и планировщик,
которым нужны параметры модели.

## Верхнеуровневые ключи

| ключ | обязателен | что делает |
|---|---|---|
| `model` | да | пайплайн, который обучаем (`_target_` + аргументы) |
| `train_dataloader` | да для обучения | `torch.utils.data.DataLoader` с датасетом и `collate_fn` |
| `valid_dataloader` | нет | если не задан, валидации не будет вообще |
| `test_dataloader` | нет | финальный прогон после обучения; обязателен для `fmlib.inference` |
| `optimizer` | да | `_partial_: True`, применяется к параметрам модели |
| `scheduler` | да | `_partial_: True`, шагает раз на границу накопления градиента |
| `train` | да | длительность, сиды, чекпоинты, ранняя остановка (см. ниже) |
| `metrics` | нет | `train_metrics` / `valid_metrics` / `test_metrics` |
| `mlflow` | да для обучения | имя эксперимента и прогона; из них же строится путь к чекпоинтам |
| `distributed`, `amp`, `ddp`, `compile` | нет | как выполнять — см. [distributed.md](distributed.md) |
| `logging` | нет | подробность логов, профайлер, метрики производительности |
| `callbacks` | нет | явный список колбэков вместо стандартного |
| `swa_model` | нет | усреднение весов (EMA/SWA) |
| `load_state` | только инференс | путь к весам для `fmlib.inference` |
| `root_dir` | нет | `chdir` в этот каталог перед запуском |

## `train:` — параметры обучения

Соответствует датаклассу `fmlib.training_arguments.TrainingArguments`.

| ключ | по умолчанию | что делает |
|---|---|---|
| `num_epochs` | `1000` | сколько эпох пройти |
| `seed` | `42` | сид; применяется одинаково на всех рангах |
| `start_epoch` | `0` | с какой эпохи начать; для продолжения обучения |
| `clip_grad_norm` | `null` | максимальная норма градиента; `null` — не обрезать |
| `max_saved_checkpoints` | `null` | сколько последних чекпоинтов хранить; `null` — все |
| `checkpoint_state` | `null` | путь к чекпоинту для полного возобновления (модель + оптимизатор + планировщик + счётчики) |
| `model_state` | `null` | путь только к весам модели; счётчики не восстанавливаются |
| `steps_before_evaluation` | `null` | валидировать раз в N шагов оптимизатора; `null` — раз в эпоху |
| `early_stopping` | `null` | блок с `_target_: fmlib.train.EarlyStopping` |
| `device_specific` | `False` | **устаревший ключ, ничего не делает** |

### `device_specific` ничего не делает

Ключ есть во всех 30 конфигах репозитория и во всех выставлен, но код его
игнорирует: он помечен deprecated и всегда трактуется как `False`. Раньше он
сдвигал сид на каждом процессе; сейчас сид одинаков на всех рангах. Можно
удалять из конфига без последствий.

### `checkpoint_state` против `model_state`

`model_state` грузит только веса — это дообучение с нуля по счётчикам.
`checkpoint_state` восстанавливает ещё и состояние оптимизатора, планировщика,
`GradScaler` и номер шага — это продолжение прерванного прогона. Вместе с ним
обычно задают `start_epoch`.

### `early_stopping`

```yaml
train:
  early_stopping:
    _target_: fmlib.train.EarlyStopping
    main_metric: roc_auc_score   # имя метрики без префикса valid_
    patience: 6                  # сколько валидаций терпеть без улучшения
    delta: 0                     # минимальное улучшение, которое считается улучшением
    strategy: max                # max | min
```

Ранняя остановка делает две вещи: останавливает обучение и **запрещает
сохранение чекпоинта**, когда метрика не улучшилась. Поэтому при включённой
ранней остановке в `best_models/` лежат только улучшавшиеся эпохи. Если блок не
задан, сохраняется каждая валидация.

Валидация, не давшая сравнимого числа — метрика отсутствует, `None` или
`nan`, — считается **отсутствием улучшения**: рекорд сохраняется, счётчик
терпения тикает, чекпоинт не пишется, а в лог идёт предупреждение. Раньше `nan`
считался улучшением, потому что любое сравнение с ним ложно: он переписывал
рекорд, обнулял счётчик и сохранял чекпоинт как лучший.

## Даталоадеры

```yaml
train_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: fmlib.data.TabularDataset
    path: /path/to/train
    shuffle_files: True
    shuffle_pq: True
  batch_size: 2048
  pin_memory: True
  drop_last: False
  num_workers: 8
  collate_fn:
    _target_: fmlib.data.TabularCollateFn
    target_column: target_attr_1
```

Это обычный `torch.utils.data.DataLoader`, поэтому доступны все его аргументы.
Важное отличие от привычного PyTorch: **датасет шардируется сам**. Никакого
`DistributedSampler` не нужно и он не должен использоваться — `TabularDataset` и
`EventSequenceDataset` делят записи между рангами и воркерами внутри себя
(`shard: True` по умолчанию).

Для валидации и теста ставьте `shuffle_files: False` и `shuffle_pq: False` —
иначе метрики будут считаться по перемешанным данным и станут невоспроизводимыми.

## `metrics:`

```yaml
metrics:
  valid_metrics:
    _target_: fmlib.metrics.ResponseMetrics
```

Три независимых ключа: `train_metrics`, `valid_metrics`, `test_metrics`. Любой
можно не указывать. Значением может быть как один объект, так и список — тогда
считаются все и их результаты объединяются.

Метрики валидации получают префикс `valid_`, теста — `test_`. Именно
беспрефиксное имя нужно писать в `early_stopping.main_metric`.

## `mlflow:`

```yaml
mlflow:
  experiment_name: tabular_hidden_states
  run_name: my_run
```

| ключ | что делает |
|---|---|
| `experiment_name` | эксперимент MLflow; входит в путь к чекпоинтам |
| `run_name` | имя прогона; тоже входит в путь к чекпоинтам |
| `tracking_uri` | адрес сервера MLflow; без него пишется в локальный `./mlruns` |

**Блок обязателен для обучения**, даже если логировать никуда не нужно: из
`experiment_name` и `run_name` строится путь к чекпоинтам, и `fmlib.train`
читает их безусловно. Само логирование выключается через `logging.enable: False`.
При инференсе блок необязателен.

### `logging_dir` не работает

Во всех 30 конфигах репозитория написано `logging_dir: /home/datalab/nfs/mlruns`,
и **этот ключ нигде не читается**. Прогоны уезжают в `./mlruns` относительно
каталога запуска. Чтобы писать в конкретное место, используйте `tracking_uri`:

```yaml
mlflow:
  tracking_uri: file:///home/datalab/nfs/mlruns
```

### Куда попадают чекпоинты

Каталог собирается из имён MLflow:
`best_models/{experiment_name}/{run_name}/{шаг}/`. Если `run_name` не равен
`debug`, а такой каталог уже существует, запуск падает на старте — это защита от
затирания чужого прогона. При отладке называйте прогон `debug`.

## `logging:`

Блок не встречается ни в одном конфиге репозитория, но читается кодом.

| ключ | по умолчанию | что делает |
|---|---|---|
| `enable` | `True` | общий выключатель логов, прогресс-бара и MLflow |
| `enable_profiler` | `False` | включает `torch.profiler` |
| `performance_metrics.enabled` | `False` | метрики производительности: пропускная способность, перекос между рангами, системные счётчики |
| `performance_metrics.sampling_interval_sec` | `2.0` | как часто снимать системные счётчики |
| `performance_metrics.native_system_metrics` | `True` | использовать встроенный сбор системных метрик MLflow |

## `swa_model:`

Усреднение весов по ходу обучения. Валидация при этом считается по усреднённой
модели.

```yaml
swa_model:
  usage: True
  _target_: torch.optim.swa_utils.AveragedModel
  _partial_: True
  avg_fn: 0.999        # коэффициент экспоненциального сглаживания
  min_num_steps: 128   # обновлять раз в N шагов оптимизатора
  min_epoch: 0         # с какой эпохи начать усреднять
```

`usage: False` отключает блок, не удаляя его. `avg_fn` — не функция, а число:
код сам строит из него экспоненциальное среднее
`avg_fn * old + (1 - avg_fn) * new`.

## `callbacks:`

Если ключа нет, собирается стандартный набор колбэков (MLflow, прогресс-бар,
статистика обучения, ранняя остановка, чекпоинты — плюс EMA, метрики
производительности и профайлер, если они включены). Явный список **полностью
заменяет** стандартный:

```yaml
callbacks:
  - _target_: fmlib.train.MLflowCallback
  - _target_: fmlib.train.ProgressBarCallback
  - _target_: fmlib.train.CheckpointCallback
    checkpoint_dir: best_models/my_run
```

Порядок важен в одном месте: колбэк ранней остановки должен идти **до**
колбэка чекпоинтов, потому что именно он запрещает сохранение.

## Конфиг инференса

`fmlib.inference` читает подмножество той же схемы:

```yaml
load_state: /path/to/model.bin   # веса; без них будет предупреждение и случайная модель
distributed:
  backend: null
amp: no
test_dataloader:
  ...
model:
  ...                            # та же архитектура, что при обучении
metrics:
  test_metrics:
    _target_: fmlib.metrics.InferenceSupervisedMetrics
    path_to_save: predicts/my_run
    save_steps: 100
    task_type: binary_clf
```

Запуск:

```bash
python -m fmlib.inference --config-dir=configs --config-name=inference
```

Блоки `train:`, `optimizer:`, `scheduler:` при инференсе не нужны. Раздел
`model:` должен точно совпадать с обучающим, иначе `load_state_dict` не сойдётся
по ключам.

О числе карт — [distributed.md](distributed.md#инференс-на-нескольких-картах-три-режима):
под `torchrun` инференс с метрикой-коллектором не выполняет ни одной
коллективной операции, каждый ранг пишет свои части, и `drop_tail: false` на
датасете сохраняет все записи.

### Необязательный `prepare_batch:`

Батч можно переписать перед скорингом. Единственный существующий случай —
кампанийный инференс, который разворачивает батч в матрицу «задача x канал»:
одна модель, один проход по данным, по предсказанию на каждую комбинацию.

```yaml
prepare_batch:
  _target_: fmlib.data.campaign.CampaignTaskChannelBatches
  tasks: ${campaign_meta.tasks}

campaign_meta:
  tasks:
    sa_response:
      comm_type: [0, 1, 2, 3]
```

Раньше кампанейский инференс был отдельной точкой входа (модуль cam_inference),
отличавшейся от обычной ровно этим циклом. Он удалён, точка входа теперь одна.

## Устаревший блок `accelerator:`

До перехода на чистый `torch.distributed` конфиги описывали запуск одним блоком
`accelerator:`. Он всё ещё читается — с `DeprecationWarning` — чтобы внешние
конфиги можно было мигрировать не разом. Таблица соответствия — в
[distributed.md](distributed.md#миграция-с-accelerator).

Если в конфиге есть и новые ключи, и `accelerator:`, побеждают новые.
