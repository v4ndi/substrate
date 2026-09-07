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
  _target_: avatar.pipeline.tabular.TabularClassification
  tabular_model:
    _target_: avatar.pipeline.tabular.TabularWithAggregatedStates
    embedding:
      _target_: avatar.nn.embedding.TabularEmbedding
      num_numerical_features: 189
      vocab_size: 172
      hidden_size: 64
    encoder:
      _target_: avatar.nn.tabular.TabularTransformer
      hidden_size: ${model.tabular_model.embedding.hidden_size}
      num_heads: 4
      num_layers: 3
    aggregation_config:
      name: linear
      num_features: 243
      emb_dim: ${model.tabular_model.embedding.hidden_size}
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
    embedding_dim: ${model.tabular_model.embedding.hidden_size}
```

Внешний вектор становится обычным токеном, и внимание работает с ним наравне с
признаками. Не забудьте увеличить `num_features` в `aggregation_config`.

**Late fusion** — после агрегации, прямо перед головой. Задаётся на пайплайне
через `extra_hidden_dim`:

```yaml
model:
  _target_: avatar.pipeline.tabular.TabularClassification
  extra_hidden_dim: 128
```

Early fusion даёт модели больше свободы, late — дешевле и устойчивее. Пример
`examples/tabular_hidden_states/` сравнивает оба на одних данных.

## Последовательностный стек

```
события ──► EventEncoder ──► backbone ──► агрегация ──► задача
```

`avatar.nn.sequential.EventEncoder` кодирует одно событие, применяя внимание по
его атрибутам; `TransformersWrapper` оборачивает backbone из HuggingFace
`transformers`. Дальше — три применения одного и того же:

* `NextKTokensPrediction` — self-supervised предобучение: предсказание
  следующих K событий;
* `SequenceModelWithAggregation` — свернуть последовательность в один вектор
  клиента (это то, что даёт `seq_hidden_state` для табличных моделей);
* `SequenceClassification` — классификация прямо по последовательности.

Типовой сценарий: предобучить `NextKTokensPrediction`, затем прогнать
`SequenceModelWithAggregation` с `model_weights` от предобучения и получить
эмбеддинги, затем подать их в табличную модель как внешние.

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

## Multi-task

Четыре пайплайна, различающиеся тем, что именно разделяется между задачами.

`MultiTaskResponse` и `MultiTaskUplift` разделяют структурно: передайте один
модуль — он будет общим для всех задач, передайте `{задача: модуль}` — у
каждой будет свой. Никакого гейтинга.

`MMoE` и `PLE` разделяют через смесь экспертов. У каждой задачи свой гейт над
пулом экспертов; в MMoE пул общий целиком, в PLE он разделён на общих
экспертов плюс приватную группу под каждую задачу. В этом и смысл PLE: гейт
задачи видит `[общие] + [свои]`, и градиент одной задачи не тянет за собой
весь пул.

`PLE` — это `MMoE` с другим backbone, тело класса пустое намеренно.

Веса гейтов выкладываются в выход как `task_gated_weights`; их читают
диагностические метрики `avatar.metrics.Entropy` и `avatar.metrics.Importance`
— по ним видно, действительно ли гейт распределяет нагрузку или схлопнулся на
одного эксперта.

## Где живёт функция потерь

Пайплайн создаёт её сам, но её можно подменить из конфига через ключ `loss:`:

```yaml
model:
  _target_: avatar.pipeline.tabular.TabularClassification
  num_classes: 2
  loss:
    _target_: avatar.losses.ClassificationLoss
    num_classes: 2
    task_type: classification
    l1_weight: 0.01
```

Одно исключение, о котором стоит знать: `NextKTokensPrediction` оставляет
головы предсказания у себя, а в модуль потерь выносит только сдвиг меток,
критерии и взвешивание. Головы содержат параметры, и их перенос переименовал бы
все ключи `lm_heads.*` в существующих чекпоинтах.

Подробнее — в руководстве по функциям потерь (см. `docs/README.md`).

## Как написать свой блок

Правило одно: **блок не считает потери**. Он преобразует представление, а
потери остаются пайплайну.

* Свой энкодер — наследуйте `avatar.nn.tabular.BaseTabularEncoder`, принимайте
  `(B, F, D)`, возвращайте `BaseTabularOutput`.
* Свою агрегацию — наследуйте `avatar.nn.utils.BaseAggregation`.
* Своё взаимодействие с воздействием — наследуйте
  `avatar.pipeline.uplift.treatment_interaction.BaseTreatmentInteraction`.

Всё это подставляется в конфиг как обычный `_target_` — регистрировать ничего
не нужно.
