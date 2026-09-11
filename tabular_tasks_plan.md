# Четыре табличные задачи на avatar_fm: план

Что нужно, чтобы `TabularTransformer` + `avatar/pipeline/{tabular,uplift}` закрывали
четыре типовые постановки — uplift, response (бинарная классификация), регрессия,
многоклассовая классификация — четырьмя конфигами, и чтобы на этом можно было
построить примеры.

Ветка: `refactor/data-sharding-hdfs`.

---

## 0. Вердикт

Каждая строка проверена запуском, а не чтением кода: собрана модель на игрушечных
данных, прогнан forward, результат отдан метрике ровно так, как это делает
`avatar.train.evaluate`.

| задача | пайплайн | collate | метрика | статус |
|---|---|---|---|---|
| **uplift** | `uplift.SLearner` | `UpliftCollateFn` | `UpliftMetrics` | **работает**, есть пример |
| **response** | `tabular.TabularClassification` | `TabularCollateFn` | `ResponseMetrics` | **не работает ни одним способом** |
| **регрессия** | `tabular.TabularClassification` | `TabularCollateFn(is_regression=True)` | `RegressionMetrics` | **запускается, но учится не тому** |
| **multi-class** | `tabular.TabularClassification` | `TabularCollateFn` | — | **метрики нет вообще** |

Ни один из трёх supervised-сценариев сегодня не доходит от конфига до числа. Это
не «нужно написать конфиги», это «нужно починить три дефекта и написать одну
метрику, и тогда конфиги станут возможны».

### Что именно ломается

**Response.** Голова на один логит даёт `(B, 1)`, а collate отдаёт таргеты `(B,)`:

```
[FAIL] binary, num_classes=1: ValueError: Target size (torch.Size([64]))
       must be the same as input size (torch.Size([64, 1]))
```

Обходной путь — `num_classes: 2` и `CrossEntropyLoss` (так написано в
`docs/getting_started.md`) — обучается, но не считается:

```
[FAIL] binary, num_classes=2 + ResponseMetrics:
       ValueError: y should be a 1d array, got an array of shape (64, 2) instead.
```

`ResponseMetrics.predictions` делает `sigmoid(logits).squeeze(1)` — она ждёт ровно
один столбец. Итого: в одной форме не учится, в другой не измеряется.

**Регрессия.** Здесь хуже, потому что молча. `MSELoss` получает `(B, 1)` и `(B,)`,
броадкастит их в `(B, B)` и считает среднюю ошибку по **всем парам**
«предсказание × чужой таргет»:

```
лосс, как считается сегодня = 2.052790
лосс по всем B×B парам      = 2.052790   ← то же самое число
лосс, который имелся в виду = 1.397575

градиент сегодня  : [ 0.530, -0.081, -0.710,  0.206, -0.345, -0.450]
градиент правильный: [ 0.379, -0.377, -0.487,  0.324, -0.163, -0.527]
```

Градиент тянет каждое предсказание к среднему таргету батча, а не к своему.
Единственный признак — `UserWarning` от torch в логе.

**Multi-class.** Лосс корректен (`CrossEntropyLoss`, `(B, K)` против `(B,)` —
проверено), но измерить нечем: в `avatar.metrics` есть `ResponseMetrics`,
`RegressionMetrics`, `UpliftMetrics` и два инференс-коллектора, и всё.

```
[FAIL] multiclass K=5 + ResponseMetrics: ValueError: multi_class must be in ('ovo', 'ovr')
```

**Uplift** работает; единственное, чего не хватает, — колонки `split_type` в
валидации, без которой калиброванные метрики не считаются (это уже
задокументировано в README примера).

---

## 1. Данные: выгрузка и препроцессинг

### 1.1 Источники

Внутренние данные (`/home/datalab/projects/avatar/...`, HDFS) на машине
недоступны, поэтому примеры строятся на открытых датасетах. Все три источника
проверены — скачиваются и читаются:

| датасет | строк | признаки | откуда | уже выгружен |
|---|---|---|---|---|
| **Hillstrom** | 64 000 | 2 числовых + 6 категориальных | `sklift.datasets.fetch_hillstrom` | да |
| **Lenta** | 687 029 | 191 числовой + 2 категориальных | `sklift.datasets.fetch_lenta` | да |
| **Covertype** | 581 012 | 10 непрерывных + 44 бинарных | `sklearn.datasets.fetch_covtype` | нет, качается |

Hillstrom и Lenta уже лежат в
`/home/jovyan/rusakov/runs/compare_training/data/` — их выгрузили для отчёта о
сравнении обучения. Скрипты `compare_training_scripts/download_data.py` и
`build_dataset.py` — рабочий прототип всего шага 1; новый скрипт делается из них.

### 1.2 Какой датасет под какую задачу

| задача | датасет | таргет | доля / диапазон | группа |
|---|---|---|---|---|
| uplift | Hillstrom → Lenta | `visit` / `response_att` | 16.7% ЦГ vs 10.6% КГ (Hillstrom) | `channel` / `main_format` |
| response | те же | тот же таргет, воздействие игнорируется | — | те же |
| регрессия | Lenta | `log1p(sale_sum_12m_g24)` | 100% непустых, 82% ненулевых, max 743 228 | `main_format` |
| multi-class | Covertype | `Cover_Type` | 7 классов, от 0.5% до 48.8% | нет |

Два замечания, которые лучше знать заранее, чем выяснить на прогоне:

* **`spend` из Hillstrom не годится под регрессию**: 99.1% нулей, ненулевых
  всего 578 строк. Взят плотный денежный таргет из Lenta. Семейство колонок
  `*_g24` при этом обязано уйти из признаков — иначе таргет предсказывается сам
  из себя.
* **Covertype несбалансирован** (класс 4 — 2 747 строк из 581 012, 0.5%, против
  48.8% у класса 2). Это аргумент за `balanced_accuracy` и macro-F1 как главные
  числа, а не за accuracy.
* **В Lenta колонка воздействия называется `group`** (значения `test` /
  `control`) — ровно то имя, под которым метрики ждут группу кампании. Скрипт
  подготовки обязан переименовать: воздействие → `treatment`, группа кампании
  (`main_format`) → `group`. Иначе всё соберётся и посчитает не то.

Каждая задача получает два масштаба: **smoke** (выборка ~20 000 строк, прогон
минуты на CPU — то, что запускается из примера) и **full** (весь датасет, A100).

### 1.3 Раскладка колонок

Общая для всех четырёх задач, чтобы конфиги отличались только там, где
отличаются задачи:

| колонка | тип | кто читает |
|---|---|---|
| `epk_id` | int64 | метрики, submit-файл |
| `target` | int64 / float32 | `collate_fn.target_column` |
| `treatment` | int64, `1 = контроль` | `UpliftCollateFn` (только uplift) |
| `group` | int64 | `GroupedPredictionMetric` — все метрики считаются по группам |
| `split_type` | string, `calib` / `test` | калибровка uplift и разделение срезов |
| `cat_features` | list\<int64\> | `TabularDataset` |
| `num_features` | list\<float32\> | `TabularDataset` |

Имена `group` и `split_type` — не косметика: `narrow_for_metrics` оставляет в
батче ровно то, что метрика объявила в `required_inputs`, а там записано
буквально `("epk_id", "group", "is_treat", "targets", "split_type")`. Колонка с
другим именем до метрики не доедет.

`group` и `epk_id` доводятся до батча через
`SupervisedCollateFn(add_extra_columns={"group": "group", "epk_id": "epk_id"})`,
`split_type` проходит как есть — строковые колонки collate не трогает, а
`move_to_device` рекурсивно возвращает строку без изменений.

### 1.4 split_type: то, чего не хватало uplift-примеру

Валидация делится пополам, стратифицированно по `(группа, воздействие, таргет)`:
`calib` — на нём подгоняются бета-калибраторы, `test` — на нём и только на нём
отчитываются калиброванные метрики. Это то, что было починено в аудите метрик
(S6/S8/S9): без колонки калибровка оценивала себя на собственных строках.

Для трёх supervised-задач `split_type` менее принципиален, но раскладка одна на
всех, и наличие двух срезов сразу даёт `calib_group_*` и `test_group_*` — видно,
насколько метрика устойчива.

### 1.5 Препроцессинг

`avatar.preprocessing.local.TabularPreprocessor`, один проход, **fit только на
train**. Дальше по книге (`docs/guides/preprocessing.md`):

```python
pp = TabularPreprocessor(
    categorical_columns=cat_cols,
    numeric_columns=num_cols,
    spec_tokens={"pad": 0},
)
pp.fit(f"{work}/train")
for split in ("train", "valid"):
    table = pp.transform(f"{work}/{split}", identity_cols=IDENTITY)
    write_shards(table, out / split, shards)
yaml.safe_dump(pp.dump(), open(f"artifacts/{name}_preprocessor.yaml", "w"))
```

Выход шардируется на 8–16 файлов, иначе файловому шардированию нечего делить
между рангами.

Скрипт обязан печатать и класть в `dims.json` три числа, без которых конфиг не
собрать:

| параметр конфига | откуда |
|---|---|
| `embedding.vocab_size` | `pp.vocab_size` |
| `embedding.num_numerical_features` | `len(num_cols)` |
| `aggregation_config.num_features` | `len(cat_cols) + len(num_cols)` (+1 за токен воздействия в uplift, +1 за токен группы) |

Для регрессии к этому добавляется трансформация таргета: `signed_log1p` и
сохранённые `mean/std`, чтобы предсказание можно было вернуть в рубли. Артефакт
кладётся рядом с препроцессором.

**Итог шага 1:** один скрипт `prepare_data.py`, четыре набора parquet, четыре
артефакта препроцессора, четыре `dims.json`.

---

## 2. Правки во фреймворке

Порядок — по тому, что блокирует.

### F1. Голова на один логит не сходится с таргетом `(B,)` — блокер response

`avatar/losses/classification.py`. Сегодня `ClassificationLoss.forward` отдаёт
`logits` и `targets` в критерий как есть. Нужна одна точка согласования формы:

```python
def align(self, logits, targets):
    """Голова шириной в один столбец предсказывает одно число на запись.

    Конфиг с ``num_classes: 1`` даёт логиты ``(B, 1)``, а collate — таргеты
    ``(B,)``. BCE на этом падает, MSE молча броадкастит в ``(B, B)``.
    """
    if logits.ndim == 2 and logits.shape[-1] == 1 and targets.ndim == 1:
        logits = logits.squeeze(-1)
    if isinstance(self.loss_fn, (nn.BCEWithLogitsLoss, nn.MSELoss)):
        targets = targets.to(logits.dtype)
    return logits, targets
```

Почему здесь, а не в пайплайне: `ClassificationLoss` — единственный слой,
который знает `num_classes` и `task_type`, а `TabularClassification` уже отдаёт
ему решение о критерии. Пользовательский `loss_fn` из конфига проходит через ту
же точку и получает согласованные формы.

Проверка: тест, который на прежней реализации падает с тем самым `ValueError`.

### F2. Регрессия учится на средний таргет батча — блокер регрессии

Чинится тем же `align`: `squeeze(-1)` убирает броадкаст. Отдельным тестом
фиксируется, что лосс равен `MSE(pred, y)`, а не среднему по парам — числа из
раздела 0 (`1.397575` против `2.052790`) годятся как эталон.

Дополнительно: в `TabularClassification.forward` стоит один
`assert`/`ValueError` с внятным текстом, если после согласования формы всё ещё
не сходятся — сейчас ошибка приходит из недр torch.

### F3. Метрики многоклассовой классификации — блокер multi-class

Новый класс в `avatar/metrics/supervised.py`, на том же скелете
`GroupedPredictionMetric`, что и остальные три:

```python
class MultiClassMetrics(GroupedPredictionMetric):
    """Качество K-классовой головы, по группам и срезам.

    Args:
        num_classes: сколько классов; нужен, чтобы ROC AUC считался по полному
            списку меток, даже если в срезе встретились не все.
        main_metric: какое число считается главным. По умолчанию
            ``balanced_accuracy`` — на несбалансированных данных accuracy
            меряет размер большого класса, а не модель.
    """

    required_inputs = ("epk_id", "group", "targets", "split_type")
    required_outputs = ("logits",)
```

Набор чисел на срез: `accuracy`, `balanced_accuracy`, `f1_macro`,
`f1_weighted`, `roc_auc_score_ovr` (macro, с явным `labels=range(K)`),
`log_loss`. Всё пропускается через `defined_scores` — срез, в котором остался
один класс, не должен выдавать `nan` (это ровно тот дефект, который чинился в
S4: `nan` стирал рекорд ранней остановки).

Два места, где многоклассовость не влезает в текущий скелет:

* **`predictions` возвращает `(N, K)`**, а не `(N,)`. `np.concatenate` по нулевой
  оси это переживает, а `write_submit` — нет: `pd.DataFrame` не принимает
  двумерную колонку. Правка в `GroupedPredictionMetric.write_submit` — разложить
  двумерную колонку в `y_pred_0 … y_pred_{K-1}`. Одно место, все наследники
  получают это бесплатно.
* **главный класс и вероятности** нужны оба: argmax для accuracy/F1, полная
  матрица для ROC AUC и log-loss. Хранить обе колонки.

### F4. MAPE на нулевых таргетах выдаёт 3.15e15

`sklearn.mean_absolute_percentage_error` при `y_true = 0` делит на
`np.finfo(float64).eps`:

```
MAPE при нулевых таргетах: 3152519739159347.0
```

Число конечное, поэтому фильтр `defined_scores` его пропускает, и оно уезжает в
MLflow, а если `main_metric = "mape"` — то и в раннюю остановку. В выбранном
регрессионном таргете нулей 18%.

Правка в `calculate_regression_metrics`: считать MAPE только по ненулевым
таргетам, писать в лог долю пропущенных, возвращать `nan` (то есть «нет числа»),
если ненулевых не осталось. Заодно добавить `rmse` и `r2` — сейчас есть только
`mse`, `mae`, `mape`.

### F5. `InferenceSupervisedMetrics` не умеет многоклассовость

`task_type` принимает `binary_clf` и `reg`; для `multi_clf` нужно писать argmax
плюс вероятности по классам. Без этого у multi-class-примера не будет
инференс-конфига, а он есть у всех остальных.

### F6. `SupervisedLearner` не переживает инференс

```
[FAIL] SupervisedLearner inference (targets=None): AttributeError: 'NoneType' object has no attribute 'device'
[OK]   SupervisedLearner train: logits=None
```

Первая строка — `_ = targets.device` в начале `forward`; вторая — логиты
считаются только в `eval`, поэтому `train_metrics` с этим пайплайном
невозможны.

Решение по существу: **не чинить, а не использовать**. Для response берётся
`TabularClassification` — он документирован, на него ссылается десяток
примеров конфигов, и он же обслуживает регрессию и multi-class, то есть три
задачи из четырёх идут одним пайплайном и отличаются тремя ключами. У
`SupervisedLearner` есть своё — эмбеддинг группы, — но ни одного конфига в
репозитории он не имеет. Предложение: починить две строки (три минуты) и
пометить в `docs/reference/pipeline.md`, что штатный путь для response — это
`TabularClassification`.

### F7. Устаревшие конфиги пилота

`experiments/sbercampaign_pilot/configs/pilot/train/*.yaml` ссылаются на
`avatar.pipeline.uplift.SLearnerExp`, которого в коде больше нет. Проверка
`tests/docs/test_config_targets.py` смотрит только `examples/` и `docs/`,
поэтому молчит. Это не блокер новых задач, но как шаблон эти файлы брать
нельзя — и об этом стоит сказать явно, потому что первым делом человек откроет
именно их.

---

## 3. Конфиги

Четыре файла, общая часть одинаковая; ниже — только то, чем они отличаются.

**Общее для всех четырёх:** `distributed`/`amp`/`ddp`, `TabularDataset` с
шардированными parquet, `TabularEmbedding` + `TabularTransformer` +
`LinearAggregation`, `AdamW` + cosine, `EarlyStopping` на главной метрике.

### 3.1 uplift

```yaml
model:
  _target_: avatar.pipeline.uplift.SLearner
  embedding: {_target_: avatar.nn.embedding.TabularEmbedding, ...}
  tabular_encoder: {_target_: avatar.nn.tabular.TabularTransformer, ...}
  aggregation_config:
    name: linear
    num_features: ${...}   # признаки + 1 за токен воздействия + 1 за токен группы
  n_groups: 4
  exchange_treatment_group: true
collate_fn:
  _target_: avatar.data.UpliftCollateFn
  target_column: target
  treatment_column: treatment
  group_column: group
  inverse_treatment: True
metrics:
  valid_metrics:
    _target_: avatar.metrics.UpliftMetrics
    require_calibration: true
```

Готово сегодня, кроме `split_type` в данных.

### 3.2 response

```yaml
model:
  _target_: avatar.pipeline.tabular.TabularClassification
  tabular_model: {_target_: avatar.pipeline.tabular.TabularWithAggregatedStates, ...}
  num_classes: 1          # один логит -> одна вероятность
  task_type: classification
collate_fn:
  _target_: avatar.data.SupervisedCollateFn
  target_column: target
  add_extra_columns: {group: group, epk_id: epk_id}
metrics:
  valid_metrics:
    _target_: avatar.metrics.ResponseMetrics
    main_metric: roc_auc_score
```

Требует **F1**.

### 3.3 регрессия

```yaml
model:
  num_classes: 1
  task_type: regression
collate_fn:
  _target_: avatar.data.SupervisedCollateFn
  target_column: target
  is_regression: True
  add_extra_columns: {group: group, epk_id: epk_id}
metrics:
  valid_metrics:
    _target_: avatar.metrics.RegressionMetrics
    main_metric: mae
train:
  early_stopping: {_target_: avatar.train.EarlyStopping, main_metric: mean_mae, strategy: min}
```

Требует **F2** и **F4**.

### 3.4 multi-class

```yaml
model:
  num_classes: 7
  task_type: classification
collate_fn:
  _target_: avatar.data.SupervisedCollateFn
  target_column: target
  add_extra_columns: {group: group, epk_id: epk_id}
metrics:
  valid_metrics:
    _target_: avatar.metrics.MultiClassMetrics
    num_classes: 7
    main_metric: balanced_accuracy
```

Требует **F3**.

К каждому — парный `inference.yaml` с `InferenceSupervisedMetrics`
(для multi-class — после **F5**).

---

## 4. Examples

Предложение: новый каталог `examples/tabular_tasks/` — четыре задачи на одном
стенде, а не четыре отдельных примера.

```
examples/tabular_tasks/
  README.md            таблица «задача -> датасет -> пайплайн -> метрика» + запуск
  prepare_data.py      выгрузка + препроцессинг всех четырёх, --task и --scale
  artifacts/           препроцессоры и dims.json (в репозиторий, они маленькие)
  configs/
    uplift.yaml        uplift_inference.yaml
    response.yaml      response_inference.yaml
    regression.yaml    regression_inference.yaml
    multiclass.yaml    multiclass_inference.yaml
```

Почему вместе: ценность именно в том, что четыре конфига стоят рядом и
отличаются тремя ключами — это видно, только если они в одном каталоге.

Существующий `examples/uplift_modeling/s_learner/` остаётся как есть: он про
внутренние данные и про late fusion с эмбеддингом последовательности, у него
другая роль. В его README добавляется ссылка на новый пример.

Формат — по правилам `examples/README.md`: `.py` плюс README, без ноутбуков в
критичном пути. `prepare_data.py` качает открытые данные сам, поэтому пример
попадает в разряд «запускается прямо сейчас» — таких сейчас только два.

---

## 5. Тесты

| что | где | зачем |
|---|---|---|
| согласование формы головы и таргета | `tests/pipeline/test_losses.py` | падает на прежней реализации (F1, F2) |
| `MultiClassMetrics`: имена, группы, срезы, вырожденный срез | `tests/metrics/test_multiclass_metrics.py` | F3 |
| MAPE при нулевых таргетах | `tests/metrics/test_grouped_metrics.py` | F4 |
| `write_submit` с двумерным `y_pred` | там же | F3 |
| четыре задачи: два шага обучения на синтетике через настоящий конфиг | `tests/pipeline/test_tabular_tasks.py` | связка «конфиг -> пайплайн -> метрика» целиком |

Конфиги примера попадут под `tests/docs/test_config_targets.py` автоматически —
он резолвит каждый `_target_` в `examples/`.

---

## 6. Порядок работ

1. **F1 + F2** — согласование формы в `ClassificationLoss`, с тестами. После
   этого response и регрессия впервые доходят от конфига до числа.
2. **F4** — MAPE и добавление `rmse`/`r2`.
3. **F3** — `MultiClassMetrics` плюс правка `write_submit` под двумерную колонку.
4. **`prepare_data.py`** — выгрузка и препроцессинг всех четырёх наборов, smoke
   и full. Самый долгий шаг по времени прогона, но не по объёму кода:
   `compare_training_scripts/build_dataset.py` покрывает две трети.
5. **Четыре train-конфига** и прогон smoke на каждом — до первых чисел.
6. **F5** — многоклассовый инференс-коллектор, четыре `inference.yaml`.
7. **README примера**, правки в `docs/reference/pipeline.md` (штатный путь для
   response), `docs/getting_started.md` (там сейчас `num_classes: 2` для того,
   что называется бинарной классификацией).
8. **Full-прогоны** на Lenta и Covertype, числа в README.
9. **F6** — две строки в `SupervisedLearner`; **F7** — решить, чинить или
   удалить конфиги пилота.

Шаги 1–3 независимы друг от друга и от шага 4: правки во фреймворке и выгрузка
данных идут параллельно.

---

## 7. Что остаётся за рамками

* **Late fusion с эмбеддингом последовательности.** Все четыре конфига строятся
  на чистых табличных признаках. `hidden_state_column` поддержан пайплайнами, но
  у открытых датасетов нет событийной истории, из которой такой эмбеддинг
  берётся. Это отдельная задача и отдельный пример.
* **Multi-task.** `avatar/pipeline/multi_task/` (MMoE, multi-task uplift и
  response) остаётся без рабочего конфига — это уже записано в отчёте по
  метрикам как открытый пункт.
* **Кастомные лоссы под дисбаланс** (focal, веса классов). Точка инъекции есть
  (`loss:` в `TabularClassification`), но подбор — за пределами «завести
  четыре постановки».
