# Справочник: `avatar.metrics`

Каталог метрик. Контракт и то, как они композируются, — в
[../guides/metrics.md](../guides/metrics.md).

Колонка «читает» показывает, какие поля выхода модели нужны метрике: это и есть
условие её применимости к конкретному пайплайну.

## Качество классификации

| класс | читает | возвращает |
|---|---|---|
| `RocAucScore` | `logits`, `targets` | `roc_auc_score`; с `group_column` — ещё по группам, их среднее и стандартное отклонение |
| `ResponseMetrics` | `logits`, `targets` | `roc_auc_score`, `recall_at_{5,10,15,20}`, `precision_at_{5,10,15,20}` |
| `RegressionMetrics` | `logits`, `targets` | `mse`, `mae`, `mape` |

`k` в `precision_at_k` и `recall_at_k` — **процент** популяции, а не число
записей.

## Uplift

| класс | читает | возвращает |
|---|---|---|
| `UpliftMetrics` | `uplift` или `treatment_probs`/`control_probs`, `treatment`, `conversion`, `group` | `uplift_at_{5,10,15,20,50}`, `uplift_auc_score`, `qini_auc_score`; с `require_calibration` — те же с префиксом `calibrated_`, плюс ROC AUC по каждой голове |

`require_calibration: True` подгоняет бета-калибратор на каждую голову.
Калибровка монотонна, поэтому не должна менять ранжирование: класс проверяет,
что ROC AUC голов сдвинулся меньше чем на 0.01, и падает по assert, если это не
так.

## Функции потерь и диагностика

| класс | читает | возвращает |
|---|---|---|
| `MultiLossMetric` | `losses`, `num_items` | по значению на компоненту составных потерь, взвешенных по числу элементов |
| `UniversalLossesMetric` | любые поля `*loss` выхода | среднее по каждому; поле `loss` логируется как `basic_loss` |
| `SequenceStats` | `seq_features` | `mean_context_length`, `event_{id}` — доли типов событий |
| `HiddensNorm` | `aggregated_hidden_state` | `l1_norm`, `l2_norm`, `l_inf_norm` |
| `Entropy` | `task_gated_weights` | `{задача}_normalized_entropy` |
| `Importance` | `task_gated_weights` | `{задача}_importance` |
| `ExpertsWorkload` | `router_logits` | **скаляр, не словарь** — в `valid_metrics` использовать нельзя, см. ниже |

`UniversalLossesMetric` находит поля рефлексией, поэтому пайплайн, у которого
появился новый вспомогательный лосс, начинает его логировать без правки
конфига.

`Entropy` и `Importance` отвечают на один вопрос с двух сторон: распределяет ли
гейт нагрузку или схлопнулся на одного эксперта. Энтропия близко к 1 — хорошо;
importance близко к 0 — хорошо.

`ExpertsWorkload` возвращает скаляр вместо словаря, а цикл делает
`scores.update(metric.compute())`. Ни в одном конфиге она не используется;
прежде чем ставить её в конфиг, поправьте тип возврата или оберните её.

## Сохранение предсказаний

Их результат — файл, а не число.

| класс | что пишет | сброс на диск |
|---|---|---|
| `ClassificationInferenceMetrics` | вероятности + указанные колонки входа | один раз, в `compute` |
| `CollectEmbeddings` | `seq_hidden_state` + дополнительные колонки | каждые `save_steps` батчей |
| `InferenceCampaignMetrics` | предсказание + кампанейские атрибуты | каждые `save_steps` |
| `InferenceMultiTaskCampaignMetrics` | вероятности обеих голов, задача, группа | каждые `save_steps` |
| `InferenceMultiTaskResponseMetrics` | одна вероятность на запись, задача | каждые `save_steps` |
| `InferenceSupervisedMetrics` | предсказание, задача, `target_attr_2` | каждые `save_steps` |

`ClassificationInferenceMetrics` — единственный из них, кто держит всё в
памяти до конца. Для прогонов по десяткам миллионов записей берите
кампанейские сборщики.

Формат имени файла — `<время>[_<prefix>].parquet`, поэтому несколько прогонов в
один каталог не затирают друг друга.

## Бенчмарки

| класс | что делает | требует |
|---|---|---|
| `CatboostCampaignBenchmark` | выгружает эмбеддинги, обучает CatBoost по каждому контуру | extra `catboost` |
| `MLPCampaignBenchmark` | выгружает эмбеддинги, запускает `avatar.train` подпроцессом по каждому контуру | конфиг обучения MLP |

Обе наследуются от сборщика: сначала дамп, потом разбиение по `report_month`
согласно `split`, потом обучение downstream-модели по каждому контуру.

Внутри метрики запускается целая обучающая задача, поэтому их место —
`test_metrics`, а не `valid_metrics`. Настройка `MLPCampaignBenchmark` разобрана
в [../mlp_benchmark/instruction.md](../mlp_benchmark/instruction.md).

## Обёртки

| класс | что делает |
|---|---|
| `GroupDevidedMetricsWrapper` | считает вложенную метрику отдельно по каждому сочетанию значений колонок |
| `GroupAverageMetricWrapper` | добавляет средние по уже посчитанным метрикам |

Вложенная метрика в `GroupDevidedMetricsWrapper` задаётся через `metric_class`
с **`_partial_: true`** — по экземпляру на группу. Без этого все группы
разделят один накопитель.

Имена, которые он производит, собираются как
`{columns_desc}_{value}_..._{имя вложенной метрики}` — например
`channel_0_group_1_roc_auc_score`. Именно по ним потом работает
`avg_over_regulars` у `GroupAverageMetricWrapper`.

## Базовые классы

| класс | для чего |
|---|---|
| `BaseMetric` | контракт `update` / `compute` / `reset` |
| `BaseInferenceMetric` | база для метрик, чей результат — файл; даёт `save_dataframe` |
