# Справочник: `avatar.pipeline`

Каталог пайплайнов — того, что ставится в `model:`. Как они собираются из
блоков — в [../guides/models.md](../guides/models.md).

Пайплайн — единственный слой, который считает функцию потерь. Всё под
`avatar/nn/` от неё свободно, поэтому одни и те же блоки переиспользуются
между задачами.

Колонка «батч» указывает, какая collate-функция должна стоять в даталоадере:
пайплайн и collate подбираются в паре.

## Табличные

| класс | батч | выход |
|---|---|---|
| `TabularClassification` | `TabularCollateFn` | `TabularOutput` |
| `TabularWithAggregatedStates` | — (вложенный блок) | тензор `(B, D)` |
| `SupervisedLearner` | `SupervisedCollateFn` | `TabularOutput` |

`TabularWithAggregatedStates` — не самостоятельный пайплайн, а
представляющий стек: `embedding(batch) -> encoder -> aggregation`. Его ставят
в `tabular_model` классификатора.

`TabularClassification` — штатный путь для трёх постановок из четырёх. Они
отличаются двумя ключами:

| постановка | `num_classes` | `task_type` | метрика |
|---|---|---|---|
| бинарная классификация (response) | `1` | `classification` | `ResponseMetrics` |
| регрессия | `1` | `regression` | `RegressionMetrics` |
| многоклассовая классификация | `K > 1` | `classification` | `MultiClassMetrics` |

`num_classes: 1` означает одно число на запись — одну вероятность или одно
значение, — и именно это читают `ResponseMetrics` и `RegressionMetrics`.
Бинарную задачу можно записать и как `num_classes: 2` с кросс-энтропией, но
тогда на выходе распределение из двух столбцов, и мерить её нужно уже
`MultiClassMetrics`. Формы головы и таргета согласует `ClassificationLoss`, так
что колонка таргета `(B,)` из любой collate-функции подходит к голове `(B, 1)`.

`TabularClassification` допускает `tabular_model: null` — тогда модель работает
только по внешним скрытым состояниям. Так устроен MLP-бенчмарк.

`SupervisedLearner` — та же форма, что у `SLearner`, но без флага воздействия:
response-постановка с опциональным эмбеддингом группы. Рабочего конфига у него
нет ни одного; для response берите `TabularClassification`.

## Uplift

| класс | батч | выход |
|---|---|---|
| `SLearner` | `UpliftCollateFn` | `MultiGroupUpliftOutput` |

Один проход на обучении, два на валидации (всё как воздействие, всё как
контроль). Отсюда `uplift = P(y|treated) - P(y|control)`.

Блоки взаимодействия с воздействием:

| класс | эффект | +1 токен |
|---|---|---|
| `ConcatTreatmentInteraction` | добавляет токен воздействия (по умолчанию) | да |
| `SumTreatmentInteraction` | прибавляет ко всем токенам | нет |
| `ElementwiseTreatmentInteraction` | умножает все токены | нет |
| `IgnoreTreatmentInteraction` | игнорирует воздействие | нет |

Аргумент `exchange_treatment_group` резервирует **последний** идентификатор
группы под контрольную: на обучении контрольные записи переразмечаются в
`n_groups - 1`, и калиброванный контрольный проход использует тот же
идентификатор.

## Чего здесь больше нет

Семейства `sequence` (`NextKTokensPrediction`, `SequenceModelWithAggregation`,
`SequenceClassification`) и `multi_task` (`MMoE`, `PLE`, `MultiTaskResponse`,
`MultiTaskUplift`) удалены из `avatar/pipeline`. Препроцессинг событийных
последовательностей и энкодеры под них (`avatar/data/sequential`,
`avatar/nn/sequential`) остались на месте — ушёл только слой задачи над ними.
Причины и границы решения — в
[../decisions/pipeline_boundaries.md](../decisions/pipeline_boundaries.md).
