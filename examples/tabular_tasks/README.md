# Четыре табличные задачи

Uplift, response, регрессия и многоклассовая классификация — на одном стенде,
четырьмя конфигами поверх `TabularTransformer`. Данные открытые, скачиваются
скриптом, так что пример запускается как есть.

Смысл в том, что конфиги стоят рядом: они отличаются двумя-тремя ключами, и
видно, какими именно.

## Навигация

* `prepare_data.py` — выгрузка и препроцессинг всех четырёх наборов;
* `configs/{uplift,response,regression,multiclass}.yaml` — обучение;
* `configs/*_inference.yaml` — инференс, по одному на задачу.

## Что где берётся

| задача | датасет | строк | таргет | пайплайн | метрика |
|---|---|---|---|---|---|
| uplift | Hillstrom | 64 000 | `visit`, с контрольной группой | `uplift.SLearner` | `UpliftMetrics` |
| response | Hillstrom | 64 000 | тот же `visit`, воздействие не учитывается | `SupervisedLearner` | `ResponseMetrics` |
| регрессия | Lenta | 687 029 | `log1p` годовых трат в одной категории | `SupervisedLearner` | `RegressionMetrics` |
| multi-class | Covertype | 581 012 | тип лесного покрова, 7 классов | `SupervisedLearner` | `MultiClassMetrics` |

Uplift и response делят один датасет намеренно: разница между «кто купит» и
«кого стоит трогать» видна, только если популяция одна и та же.

Регрессия берёт Lenta, а не колонку `spend` из Hillstrom, которая кажется
очевидным выбором: в ней 99.1% нулей и всего 578 ненулевых записей из 64 000.
Взятая вместо неё колонка плотная — 82% ненулевых — и с тяжёлым хвостом, отсюда
`log1p`. Все 16 колонок того же семейства выброшены из признаков: это агрегаты
по той же категории за тот же год, и с ними модель читает ответ со своего входа.

## Запуск

```bash
# из корня репозитория

# 1. данные: ~20 000 строк на задачу, минуты на CPU
python examples/tabular_tasks/prepare_data.py --task all --scale smoke

# 2. обучение
python -m avatar.train --config-dir=examples/tabular_tasks/configs --config-name=uplift
python -m avatar.train --config-dir=examples/tabular_tasks/configs --config-name=response
python -m avatar.train --config-dir=examples/tabular_tasks/configs --config-name=regression
python -m avatar.train --config-dir=examples/tabular_tasks/configs --config-name=multiclass

# 3. инференс: предсказания лягут в predict/<задача>/
python -m avatar.infer --config-dir=examples/tabular_tasks/configs --config-name=response_inference
```

Многокарточный запуск — та же команда под `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 -m avatar.train \
    --config-dir=examples/tabular_tasks/configs --config-name=multiclass
```

Две мелочи, о которые легко споткнуться. Каталог чекпоинтов
(`best_models/tabular_tasks/<run_name>`) не перезаписывается: перед повторным
прогоном его нужно удалить или сменить `run_name`. И `load_state` в
инференс-конфиге указывает на конкретный шаг — чекпоинт пишется на каждом
улучшении, поэтому нужен последний сохранённый, а его номер зависит от того,
где сработала ранняя остановка.

### Полный масштаб

Данные строятся тем же скриптом с `--scale full`, а конфиг править не нужно:
любой ключ переопределяется из командной строки.

```bash
python examples/tabular_tasks/prepare_data.py --task regression --scale full

D=examples/tabular_tasks/data
python -m avatar.train --config-dir=examples/tabular_tasks/configs \
    --config-name=regression \
    train_dataloader.dataset.path=$D/regression_full/train \
    valid_dataloader.dataset.path=$D/regression_full/valid \
    mlflow.run_name=regression_full
```

Одну размерность всё же надо передать отдельно, и это не формальность: словарь
категорий зависит от выборки. У multi-class на всей выборке `vocab_size` равен
133 против 132 на smoke — одно значение просто не попало в подвыборку, — так что
к команде добавляется
`model.embedding.vocab_size=133`. Точные числа печатает
`prepare_data.py`.

## Что получается

Десять эпох, ранняя остановка с `patience: 3`, модель из двух слоёв шириной 64.
В колонках — лучшее значение главной метрики за прогон.

| задача | главная метрика | smoke (~20k) | full |
|---|---|---|---|
| uplift | `mean_calibrated_qini_auc_score` | 0.081 | 0.068 |
| response | `mean_roc_auc_score` | 0.647 | 0.632 |
| регрессия | `mean_mae`, шкала `log1p` (меньше — лучше) | 1.92 | **1.14** |
| multi-class | `mean_balanced_accuracy` | 0.385 | **0.575** |

Uplift-строку не стоит читать как точное число: на smoke-выборке та же
конфигурация с сидами 1, 2, 3 даёт 0.082, 0.069 и 0.059, а с сидом из конфига —
0.099. Qini на двадцати тысячах записей — шумная величина, и разброс здесь
больше любой разницы, которую на этих данных можно было бы объяснить моделью.

Две задачи, у которых smoke-выборка была в тридцать раз меньше полной, от
полных данных заметно выигрывают. Две другие живут на Hillstrom, где и полный
набор — 64 000 строк, поэтому разницы почти нет: smoke там и есть почти весь
датасет.

Это по-прежнему числа отладочного масштаба. Они показывают, что связка
«конфиг → пайплайн → метрика» работает и что данные в неё приходят
осмысленные, а не то, чего можно добиться на этих датасетах.

## Общая раскладка данных

Все четыре набора пишутся одинаково, поэтому конфиги и отличаются только там,
где отличаются задачи:

| колонка | кто читает |
|---|---|
| `epk_id` | метрики, файл предсказаний |
| `target` | `collate_fn.target_column` |
| `treatment` | `UpliftCollateFn`, только uplift; `1 = контроль` |
| `group` | все метрики считаются по группам |
| `split_type` | `calib` / `test` внутри валидации |
| `cat_features`, `num_features` | `TabularDataset` |

Имена `group` и `split_type` не свободные. Метрики объявляют их в
`required_inputs`, а цикл оценки отправляет на ранг 0 ровно объявленные ключи —
колонка с другим именем до метрики не доедет. Поэтому в supervised-конфигах
стоит `add_extra_columns: {group: group}`: метрике нужен тензор, а не список.

`split_type` — то, чего не хватало прежнему uplift-примеру. Валидация делится
пополам стратифицированно: на `calib` подгоняются бета-калибраторы, на `test`
и только на нём отчитываются калиброванные метрики. Без этой колонки калибровка
оценивала бы себя на собственных строках, поэтому её просто не считают.

## Чем отличаются конфиги

Uplift — единственный, у которого своя collate-функция и свой пайплайн:

```yaml
collate_fn:
  _target_: avatar.data.UpliftCollateFn
  treatment_column: treatment
  group_column: group
  inverse_treatment: True
model:
  _target_: avatar.pipeline.tabular.SLearner
  n_groups: 4                # 3 канала плюс идентификатор под контроль
  exchange_treatment_group: true
  aggregation_config:
    num_features: 10         # 8 признаков + токен воздействия + токен группы
```

Остальные три — один и тот же `SupervisedLearner`, и различие целиком в
трёх строчках:

| | response | регрессия | multi-class |
|---|---|---|---|
| `num_classes` | `1` | `1` | `7` |
| `task_type` | `classification` | `regression` | `classification` |
| `is_regression` в collate | — | `True` | — |
| метрика | `ResponseMetrics` | `RegressionMetrics` | `MultiClassMetrics` |
| `main_metric` | `mean_roc_auc_score` | `mean_mae`, `strategy: min` | `mean_balanced_accuracy` |

`num_classes: 1` означает одно число на запись. Голова выдаёт `(B, 1)`, а
collate — `(B,)`; согласует их `ClassificationLoss`.

`mean_balanced_accuracy`, а не accuracy: в Covertype на класс 2 приходится
48.8% записей, а на класс 4 — 0.5%, и обычная accuracy мерила бы размер
большого класса. В smoke-прогоне это видно прямо: accuracy 0.64 при
balanced accuracy 0.29.

## Три числа для конфига

`prepare_data.py` печатает их в конце и кладёт в dims.json рядом с данными:

| параметр | uplift | response | регрессия | multi-class |
|---|---|---|---|---|
| `embedding.vocab_size` | 26 | 26 | 8 | 132 |
| `embedding.num_numerical_features` | 2 | 2 | 175 | 10 |
| `aggregation_config.num_features` | 10 | 8 | 177 | 54 |

Для uplift к числу признаков прибавляются два токена: воздействия и группы.

## Куда дальше

* [../../docs/reference/pipeline.md](../../docs/reference/pipeline.md) — какой пайплайн под какую постановку;
* [../../docs/reference/metrics.md](../../docs/reference/metrics.md) — что именно считает каждая метрика;
* [../../docs/guides/preprocessing.md](../../docs/guides/preprocessing.md) — препроцессинг подробно;
* [../uplift_modeling/s_learner/](../uplift_modeling/s_learner/) — uplift на внутренних данных, с эмбеддингом последовательности клиента.
