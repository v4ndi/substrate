# Справочник: `fmlib.pipeline`

Каталог пайплайнов — того, что ставится в `model:`. Как они собираются из
блоков — в [../guides/models.md](../guides/models.md).

Пайплайн — единственный слой, который считает функцию потерь. Всё под
`fmlib/nn/` от неё свободно, поэтому одни и те же блоки переиспользуются
между задачами.

Колонка «батч» указывает, какая collate-функция должна стоять в даталоадере:
пайплайн и collate подбираются в паре.

## Табличные

| класс | батч | выход |
|---|---|---|
| `SupervisedLearner` | `SupervisedCollateFn` / `TabularCollateFn` | `TabularOutput` |

Один класс на три постановки; различаются они двумя ключами:

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

Пайплайн владеет всем табличным стеком: эмбеддинг, энкодер, агрегация, внешние
эмбеддинги (late fusion), голова, функция потерь. Отдельного класса под
представление больше нет — он существовал только затем, чтобы его можно было
переиспользовать, а переиспользовать его было некому: два других пайплайна
собирали стек у себя внутри.

`embedding: null` вместе с `tabular_encoder: null` оставляет модель работать
только по внешним скрытым состояниям — так устроен MLP-бенчмарк.

Необязательный эмбеддинг группы (`n_groups`) добавляет токен к признакам, ровно
как в uplift; `group_interaction` решает, каким способом.

## Uplift

| класс | батч | выход |
|---|---|---|
| `SLearner` | `UpliftCollateFn` | `MultiGroupUpliftOutput` |

`SLearner` **наследует** `SupervisedLearner` — это и есть определение
S-Learner'а: та же модель, только флаг воздействия подан как признак. Он
принимает все аргументы родителя и добавляет к ним `separate_heads`,
`treatment_interaction`, `calculate_train_uplift`, `exchange_treatment_group` и
`loss_fn`.

Один проход на обучении, два на валидации (всё как воздействие, всё как
контроль). Отсюда `uplift = P(y|treated) - P(y|control)`.

Отличий в устройстве два, и оба зафиксированы существующими чекпоинтами:
голова шириной 2 вместо `num_classes`, и в ней другой порядок слоёв —
dropout первым, активация перед нормализацией. Свою функцию потерь `SLearner`
берёт из ключа `loss_fn:`, а не `loss:`, и вызывает иначе; почему так и что с
этим делать — в
[../decisions/pipeline_boundaries.md](../decisions/pipeline_boundaries.md).

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
`MultiTaskUplift`) удалены из `fmlib/pipeline`. Препроцессинг событийных
последовательностей и энкодеры под них (`fmlib/data/sequential`,
`fmlib/nn/sequential`) остались на месте — ушёл только слой задачи над ними.
Причины и границы решения — в
[../decisions/pipeline_boundaries.md](../decisions/pipeline_boundaries.md).
