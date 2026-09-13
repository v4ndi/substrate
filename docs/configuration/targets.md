# Каталог `_target_`

Классы, которые собираются из конфига. Это и есть реальный публичный API
библиотеки: конфиги репозитория используют около трёх десятков имён, а не
несколько сотен функций.

Числа в скобках — сколько раз таргет встречается в YAML репозитория (без
`experiments/`). Они показывают, что действительно ходовое, а что редкое.

Схема конфига целиком — в [schema.md](schema.md).

---

## Датасеты

Оба датасета читают parquet, шардируются по записям между рангами и воркерами и
не требуют `DistributedSampler`.

### `avatar.data.TabularDataset` (21)

Табличные признаки построчно.

```yaml
dataset:
  _target_: avatar.data.TabularDataset
  path: /path/to/train           # файл, каталог или список путей
  shuffle_files: True            # перемешивать порядок файлов
  shuffle_pq: True               # перемешивать строки внутри файла
  hidden_state_column: seq_hidden_state   # колонка с внешним эмбеддингом
```

Основные аргументы:

| аргумент | по умолчанию | что делает |
|---|---|---|
| `path` | — | путь или список путей к parquet |
| `read_columns` | `None` | читать только эти колонки; заметно ускоряет чтение |
| `shuffle_files` / `shuffle_pq` | `False` / `True` | перемешивание; для валидации ставьте оба в `False` |
| `hidden_state_column` | `None` | колонка с внешним эмбеддингом клиента |
| `hidden_state_columns` | `None` | то же, но несколько колонок |
| `sampler` | `None` | фильтрация записей, см. ниже |
| `filesystem` | `None` | явные параметры подключения (HDFS) |
| `shard` | `True` | делить записи между рангами |
| `drop_tail` | `True` | отбрасывать хвост, не делящийся между рангами |
| `rotate_tail` | `False` | вместо отбрасывания сдвигать хвост по эпохам |
| `scan_workers` | `4` | параллелизм фазы сканирования |
| `filter_cache` | `True` | кэшировать результат сканирования с фильтром |

### `avatar.data.EventSequenceDataset` (6)

Событийные последовательности: каждая строка parquet — клиент со списками
событий.

```yaml
dataset:
  _target_: avatar.data.EventSequenceDataset
  path: /path/to/train
  sequence_columns: [mcc, price]
  event_time_column: evt_dttm
  min_length: 1
  max_length: 512
  random_slicing: False
```

Дополнительно к общим аргументам: `sequence_columns` (обязателен),
`event_time_column`, `event_ids_column`, `selected_event_ids`, `min_length`,
`max_length`, `random_slicing`, `has_tabular`.

### Сэмплеры

Отбирают подмножество записей на фазе сканирования, до чтения полезной нагрузки.

* `avatar.data.sampler.ColumnFilterSampler` — фильтр по колонке:
  `column`, `min_value`, `max_value`, `allowed_values`.
* `avatar.data.sampler.MultiTaskColumnsFilterSampler` — то же для нескольких
  колонок в multi-task постановке.

---

## Collate-функции

Собирают список записей в батч. Выбор collate определяет, какой батч-объект
получит модель, поэтому он должен соответствовать пайплайну.

| таргет | использований | для чего |
|---|---|---|
| `avatar.data.TabularCollateFn` | 18 | обычная табличная классификация/регрессия |
| `avatar.data.EventSequenceCollateFn` | 6 | событийные последовательности |
| `avatar.data.UpliftCollateFn` | 3 | uplift: добавляет флаг воздействия |
| `avatar.data.SupervisedCollateFn` | — | классификация с дополнительными колонками |
| `avatar.data.MultiTaskSupervisedCollateFn` | — | multi-task: добавляет идентификатор задачи |
| `avatar.data.MultiTaskUpliftCollateFn` | — | multi-task + uplift |

Ходовые аргументы:

```yaml
collate_fn:
  _target_: avatar.data.TabularCollateFn
  target_column: target_attr_1   # колонка целевой переменной
  is_regression: False           # True -> таргет float, а не long
```

`UpliftCollateFn` дополнительно принимает `treatment_column` и
`inverse_treatment` (поменять местами 0 и 1, если в данных 1 — контрольная
группа).

`EventSequenceCollateFn` принимает `sequence_columns` (обязателен),
`create_attention_mask`, `has_tabular`, `length_to_pad`.

---

## Эмбеддинги

### `avatar.nn.embedding.TabularEmbedding` (10)

Отображает табличные признаки в `(B, F, D)` — по вектору на признак.

```yaml
embedding:
  _target_: avatar.nn.embedding.TabularEmbedding
  num_numerical_features: 189   # сколько числовых признаков
  vocab_size: 172               # размер словаря категориальных
  hidden_size: 64               # D
  std_noise: null               # шум к числовым признакам при обучении
  hidden_state_aggregator: ...  # как подмешать внешний эмбеддинг
```

### `avatar.nn.embedding.EventSequenceEmbedding` (3)

То же для событий: `hidden_size`, `columns_meta` (описание колонок — тип и
размер словаря), `std_noise`.

### Подмешивание внешнего эмбеддинга

Оба класса принимают `hidden_state_dim` и `embedding_dim`:

* `avatar.nn.embedding.LayerNormConcatenate` (4) — нормализовать и добавить
  внешний эмбеддинг **как ещё один признак** (early fusion). Не забудьте
  увеличить `num_features` в `aggregation_config` на единицу.
* `avatar.nn.embedding.LayerNormSum` — нормализовать и прибавить ко всем
  эмбеддингам признаков.

---

## Энкодеры

### `avatar.nn.tabular.TabularTransformer` (10)

Трансформер по токенам-признакам: `(B, F, D) -> (B, F, D)`.

```yaml
encoder:
  _target_: avatar.nn.tabular.TabularTransformer
  hidden_size: 64
  num_heads: 4
  num_layers: 3
  attn_dropout: 0.15
```

Принимает уже посчитанные эмбеддинги и ничего не знает о том, откуда они
взялись. Наследуется от `avatar.nn.tabular.BaseTabularEncoder` — от него же
наследуйте свой энкодер, если пишете собственный.

### `avatar.nn.tabular.MLPEmbedding` (1)

Узкоспециальный модуль: эмбеддит две категориальные кампанейские колонки в один
плоский вектор — вход для downstream-модели поверх выгруженных эмбеддингов.

### `avatar.nn.sequential.EventEncoder` (3)

Кодирует событие с вниманием по его атрибутам: `embedding`, `dropout_p`,
`pos_embedding`, `time_encoding` (`absolute` | `delta`), `id_embedding`,
`aggregation_mode`.

### `avatar.nn.sequential.TransformersWrapper` (3)

Обёртка над backbone из HuggingFace `transformers`: `event_encoder`,
`backbone`, `output_hidden_states`.

---

## Пайплайны

Пайплайн — то, что стоит в `model:`. Он связывает блоки и считает функцию потерь.

### `avatar.pipeline.tabular.SupervisedLearner` (6)

Обучение с учителем по табличным данным: бинарная классификация, регрессия и
многоклассовая — различаются только `num_classes` и `task_type`.

```yaml
model:
  _target_: avatar.pipeline.tabular.SupervisedLearner
  embedding: ...                # avatar.nn.embedding.TabularEmbedding
  tabular_encoder: ...          # avatar.nn.tabular.TabularTransformer
  aggregation_config:
    name: linear                # sum | sum_layernorm | mean | last | linear | conv
    num_features: 243           # столько токенов приходит на агрегацию
    emb_dim: 64
  num_classes: 1                # 1 -> одно число на запись; K > 1 -> распределение
  task_type: classification     # classification | regression
  dropout_p: 0.15
  out_head_hidden_dim: 256      # по умолчанию — ширина входа головы
  hidden_state_dim: null        # late fusion: ширина внешнего эмбеддинга
  proj_hiddens_to_dim: null     # проецировать его, а не только нормировать
  n_groups: null                # эмбеддинг группы как ещё один токен
```

`aggregation_config` — не `_target_`, а словарь, который разбирает
`avatar.nn.utils.get_aggregation_layer`. Имя выбирает класс агрегации,
остальные ключи уходят в его конструктор; `linear` требует `num_features` и
`emb_dim`, `last` не требует ничего.

`num_features` — число токенов **на входе агрегации**, а не признаков в
данных. Каждый добавляющий токен механизм увеличивает его на единицу:
`n_groups`, признак воздействия в `SLearner`, внешний эмбеддинг через
`LayerNormConcatenate`.

`embedding: null` вместе с `tabular_encoder: null` — допустимо: тогда модель
работает только по внешним скрытым состояниям (так устроен MLP-бенчмарк), и
`hidden_state_dim` обязателен.

### `avatar.pipeline.tabular.SLearner` (48)

Uplift в постановке S-Learner: тот же `SupervisedLearner`, но признак
воздействия подаётся в модель наравне с остальными, голова шириной 2, а на
валидации батч прогоняется дважды. Принимает всё перечисленное выше плюс
`separate_heads`, `treatment_interaction`, `calculate_train_uplift`,
`exchange_treatment_group` и `loss_fn`.

`avatar.pipeline.tabular.IgnoreTreatmentInteraction` — заглушка взаимодействия
с воздействием, без параметров.

Старый путь `avatar.pipeline.uplift.*` ещё работает и предупреждает об
устаревании; он будет удалён в следующем релизе.

---

## Метрики

Подробнее — в разделе метрик; здесь только то, что ставят в конфиг.

| таргет | что считает |
|---|---|
| `avatar.metrics.UpliftMetrics` | uplift@k, qini, калиброванные варианты и диагностика калибровки |
| `avatar.metrics.ResponseMetrics` | ROC AUC, precision@k и recall@k в кампанейской постановке |
| `avatar.metrics.RegressionMetrics` | MSE / MAE / MAPE |
| `avatar.metrics.MultiLossMetric` | компоненты составной функции потерь |
| `avatar.metrics.UniversalLossesMetric` | все поля `*loss` выхода, найденные рефлексией |
| `avatar.metrics.CollectEmbeddings` | выгружает эмбеддинги в parquet |
| `avatar.metrics.InferenceMultiTaskCampaignMetrics` | выгружает вероятности обеих голов при инференсе кампании |
| `avatar.metrics.InferenceSupervisedMetrics` | выгружает предсказание на запись при инференсе |

### Обёртки

Метрики композируются. Обёртки — самый частый источник непонимания в конфигах,
поэтому вот что они делают:

* `avatar.metrics.utils.GroupDevidedMetricsWrapper` (4) — считает **одну и ту
  же** метрику отдельно по каждому сочетанию значений заданных колонок.
  `columns_to_devide: [target_attr_2, target_attr_3]` и
  `columns_desc: [channel, group]` дадут имена вида
  `channel_0_group_1_roc_auc_score`. Вложенная метрика указывается через
  `metric_class` с `_partial_: true` — по одному экземпляру на группу.
* `avatar.metrics.utils.GroupAverageMetricWrapper` (8) — усредняет уже
  посчитанные метрики. `avg_over_regulars` усредняет по регулярному выражению
  над именами, `groups` — по явному списку имён.

Типичная конструкция — усреднение по группам, а затем усреднение усреднений:

```yaml
metrics:
  valid_metrics:
    - _target_: avatar.metrics.utils.GroupAverageMetricWrapper
      metric:
        _target_: avatar.metrics.utils.GroupDevidedMetricsWrapper
        columns_to_devide: [target_attr_2, target_attr_3]
        columns_desc: [channel, group]
        metric_class:
          _target_: avatar.metrics.ResponseMetrics
          _partial_: true          # обязательно: по экземпляру на группу
      avg_over_regulars:
        avg_control_roc_auc: ^channel_\d+_group_1_.*roc_auc_score
        avg_target_roc_auc: ^channel_\d+_group_0_.*roc_auc_score
```

---

## Функции потерь

Обычно пайплайн создаёт функцию потерь сам; задавать её в конфиге нужно, только
если хочется другую. Все они принимают ключ `loss:` соответствующего пайплайна.

| таргет | что считает |
|---|---|
| `avatar.losses.ClassificationLoss` | MSE / BCE / CrossEntropy по `task_type`, плюс L1-регуляризация |
| `avatar.losses.CompositeLoss` | взвешенная сумма нескольких функций потерь |
| `avatar.losses.KLDLoss`, `ContrastiveLoss`, `ResearchLosses` | исследовательские функции потерь |

---

## Обучение

### `avatar.train.EarlyStopping` (7)

```yaml
early_stopping:
  _target_: avatar.train.EarlyStopping
  main_metric: roc_auc_score
  patience: 10
  delta: 0
  strategy: max      # max | min
```

### Колбэки

Ставятся в `callbacks:` и заменяют стандартный набор целиком:
`MLflowCallback`, `ProgressBarCallback`, `CheckpointCallback`,
`EarlyStoppingCallback`, `EMACallback`, `TrainStatsCallback`,
`TrainMetricsCallback`, `PerfMetricsCallback`, `ThroughputCallback`,
`ProfilerCallback` — все из `avatar.train`.

---

## Не из avatar

В конфигах встречаются и внешние таргеты; они работают так же:

* `torch.utils.data.DataLoader` — даталоадеры;
* `torch.optim.AdamW` и другие оптимизаторы, с `_partial_: True`;
* `transformers.optimization.get_scheduler` — планировщик, с `_partial_: True`;
* `torch.optim.swa_utils.AveragedModel` — усреднение весов.
