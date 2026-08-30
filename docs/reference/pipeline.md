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

`TabularClassification` допускает `tabular_model: null` — тогда модель работает
только по внешним скрытым состояниям. Так устроен MLP-бенчмарк.

`SupervisedLearner` — та же форма, что у `SLearner`, но без флага воздействия:
response-постановка с опциональным эмбеддингом группы.

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

## Последовательностные

| класс | батч | выход |
|---|---|---|
| `NextKTokensPrediction` | `EventSequenceCollateFn` | `SequenceOutput` |
| `SequenceModelWithAggregation` | `EventSequenceCollateFn` | `BaseSequenceOutput` |
| `SequenceClassification` | `EventSequenceCollateFn` | `SequenceOutput` |

`NextKTokensPrediction` — предобучение: по одной голове на признак, на каждый
шаг горизонта. Горизонты взвешиваются как `loss_k / k ** horizion_loss_weight`;
ноль означает равный вес.

`SequenceModelWithAggregation` — без головы и без потерь: даёт эмбеддинг
клиента для табличных моделей. Аргумент `model_weights` загружает результат
предобучения.

## Multi-task

| класс | как разделяет | выход |
|---|---|---|
| `MultiTaskResponse` | структурно: общий модуль либо словарь по задачам | `MultiTaskxGroupResponseOutput` |
| `MultiTaskUplift` | то же, плюс двухпроходное скоринговое поведение S-Learner | `MultiTaskxGroupUpliftOutput` |
| `MMoE` | гейт на задачу над общим пулом экспертов | `MMoEOutput` |
| `PLE` | гейт видит `[общие эксперты] + [свои]` | `MMoEOutput` |

Вспомогательные классы:

| класс | роль |
|---|---|
| `MMoEBackbone` | пул экспертов плюс гейт на задачу |
| `PLEBackbone` | пул, разделённый на общих и приватных экспертов |
| `TaskHead` | голова одной задачи; размер выставляется пайплайном через `init_head` |
| `MLPExpert` | дешёвый эксперт: остаточный feed-forward блок |
| `TabBackboneExpert` | эксперт целиком из табличного энкодера |
| `HierarchicalFeatureGate` | обучаемый гейт по признакам: общая и задачная компоненты |

`PLE` — `MMoE` с `PLEBackbone`; тело класса пустое намеренно, отдельное имя
нужно, чтобы называть архитектуру в конфигах и чтобы PLE-специфичные метрики
узнавали прогон.

Число экспертов в PLE — `num_shared_experts + num_tasks * num_specific_experts`:
приватные эксперты считаются **на задачу**.

Веса гейтов доступны в выходе как `task_gated_weights`; их читают
`avatar.metrics.Entropy` и `avatar.metrics.Importance`.
