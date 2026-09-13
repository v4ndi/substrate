# Из чего собирается модель

В конфиге стоит один объект — `model:`. Это **пайплайн**: он владеет
композицией блоков и считает функцию потерь. Всё, что лежит под `avatar/nn/`,
функцию потерь не считает вообще — именно поэтому одни и те же блоки
переиспользуются между задачами.

```
батч ──► эмбеддинг ──► энкодер ──► агрегация ──► голова ──► функция потерь
         avatar/nn      avatar/nn   avatar/nn    пайплайн    avatar/losses
```

Каталог классов — в [../reference/pipeline.md](../reference/pipeline.md) и
[../configuration/targets.md](../configuration/targets.md).

## Табличный стек

Три слоя, каждый заменяем независимо:

```yaml
model:
  _target_: avatar.pipeline.tabular.SupervisedLearner
  embedding:
    _target_: avatar.nn.embedding.TabularEmbedding
    num_numerical_features: 189
    vocab_size: 172
    hidden_size: 64
  tabular_encoder:
    _target_: avatar.nn.tabular.TabularTransformer
    hidden_size: ${model.embedding.hidden_size}
    num_heads: 4
    num_layers: 3
  aggregation_config:
    name: linear
    num_features: 243
    emb_dim: ${model.embedding.hidden_size}
  num_classes: 2
  task_type: classification
```

**Эмбеддинг** превращает батч в `(B, F, D)` — по вектору на признак.
**Энкодер** контекстуализирует эти токены: `(B, F, D) -> (B, F, D)`. Он
принимает уже посчитанные эмбеддинги и ничего не знает об их происхождении,
поэтому энкодер и эмбеддинг — соседние поля, а не вложенные. **Агрегация**
сворачивает токены в один вектор `(B, D)`. **Голова** проецирует его в число
классов и считает потери.

Три размерности, которые чаще всего ставят неверно:

| параметр | откуда берётся |
|---|---|
| `vocab_size` | `pp.vocab_size` препроцессора |
| `num_numerical_features` | длина `num_features` после препроцессинга |
| `aggregation_config.num_features` | сколько токенов приходит на агрегацию |

Последнее — не число признаков в данных, а число токенов **на входе
агрегации**. Каждый механизм, добавляющий токен, увеличивает его на единицу:
внешний эмбеддинг через `LayerNormConcatenate`, признак воздействия в uplift,
эмбеддинг группы.

## Внешние эмбеддинги: early и late fusion

Готовый эмбеддинг клиента (например, `seq_hidden_state`, полученный
последовательностной моделью) подмешивается двумя способами.

**Early fusion** — как ещё один признак, до энкодера. Задаётся внутри
эмбеддинга:

```yaml
embedding:
  _target_: avatar.nn.embedding.TabularEmbedding
  hidden_state_aggregator:
    _target_: avatar.nn.embedding.LayerNormConcatenate
    hidden_state_dim: 128
    embedding_dim: ${model.embedding.hidden_size}
```

Внешний вектор становится обычным токеном, и внимание работает с ним наравне с
признаками. Не забудьте увеличить `num_features` в `aggregation_config`.

**Late fusion** — после агрегации, прямо перед головой. Задаётся на пайплайне
через `hidden_state_dim`:

```yaml
model:
  _target_: avatar.pipeline.tabular.SupervisedLearner
  hidden_state_dim: 128
  proj_hiddens_to_dim: 64 # необязательно: проецировать, а не только нормировать
```

Early fusion даёт модели больше свободы, late — дешевле и устойчивее. Выбор
между ними — вопрос того, где живут внешние эмбеддинги: до агрегации или после
неё.

## Последовательностный стек

```
события ──► EventEncoder ──► backbone ──► агрегация ──► (пайплайна нет)
```

`avatar.nn.sequential.EventEncoder` кодирует одно событие, применяя внимание по
его атрибутам; `TransformersWrapper` оборачивает backbone из HuggingFace
`transformers`.

Пайплайна над ними сейчас нет: последовательностные и multi-task пайплайны
удалены — см.
[../decisions/pipeline_boundaries.md](../decisions/pipeline_boundaries.md).
Энкодеры, препроцессинг и `EventSequenceBatch` остались; не хватает только слоя
задачи, который взял бы у них представление и посчитал по нему потери.

## Uplift

`SLearner` — «одна модель, флаг воздействия как признак». На обучении батч
проходит один раз с настоящим значением флага. **На валидации — дважды**: всё
как воздействие, всё как контроль, и uplift равен разности вероятностей.
Отсюда две особенности: валидация примерно вдвое дороже обычной, а
`calculate_train_uplift` по умолчанию выключен.

Как флаг попадает в представление — сменный блок:

| класс | что делает | меняет число токенов |
|---|---|---|
| `ConcatTreatmentInteraction` | добавляет токен воздействия | да, +1 |
| `SumTreatmentInteraction` | прибавляет ко всем токенам | нет |
| `ElementwiseTreatmentInteraction` | умножает все токены | нет |
| `IgnoreTreatmentInteraction` | игнорирует | нет |

По умолчанию — конкатенация, поэтому `num_features` в uplift-конфигах на
единицу больше числа признаков (и ещё на единицу больше, если задан
`n_groups`).

## Где живёт функция потерь

Пайплайн создаёт её сам, но её можно подменить из конфига через ключ `loss:`:

```yaml
model:
  _target_: avatar.pipeline.tabular.SupervisedLearner
  num_classes: 2
  loss:
    _target_: avatar.losses.ClassificationLoss
    num_classes: 2
    task_type: classification
    l1_weight: 0.01
```

Подробнее — в руководстве по функциям потерь (см. `docs/README.md`).

## Как написать свой блок

Правило одно: **блок не считает потери**. Он преобразует представление, а
потери остаются пайплайну.

* Свой энкодер — наследуйте `avatar.nn.tabular.BaseTabularEncoder`, принимайте
  `(B, F, D)`, возвращайте `BaseTabularOutput`.
* Свою агрегацию — наследуйте `avatar.nn.utils.BaseAggregation`.
* Своё взаимодействие с воздействием — наследуйте
  `avatar.pipeline.tabular.interaction.BaseTreatmentInteraction`.

Всё это подставляется в конфиг как обычный `_target_` — регистрировать ничего
не нужно.
