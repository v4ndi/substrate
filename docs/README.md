# Документация avatar

Навигация по документации репозитория. Всё, что не является кодом и примерами,
лежит здесь.

## С чего начать

[getting_started.md](getting_started.md) — путь от установки до инференса.
Корневой [README.md](../README.md) отвечает на вопросы «что это» и «как
установить», а [examples/](../examples/README.md) содержит сквозные примеры.

## Конфигурация

Обучение и инференс целиком описываются одним YAML, поэтому это главный раздел
для того, кто пользуется библиотекой, а не дорабатывает её.

* [configuration/schema.md](configuration/schema.md) — все ключи конфига, их
  значения по умолчанию и то, чего они **не** делают.
* [configuration/targets.md](configuration/targets.md) — каталог классов,
  которые можно подставить в `_target_`.
* [configuration/distributed.md](configuration/distributed.md) — multi-GPU и
  несколько узлов, смешанная точность, DDP, миграция с `accelerator:`.

## Руководства

Как устроена подсистема и как её расширить.

* [guides/training.md](guides/training.md) — цикл обучения: эпохи против шагов,
  валидация, чекпоинты, возобновление, накопление градиента.
* [guides/callbacks.md](guides/callbacks.md) — двенадцать событий, контракт
  контекста, единственное ограничение на порядок, как написать свой.
* [guides/losses.md](guides/losses.md) — `Loss` / `LossOutput`, подмена из
  конфига, композиция, когда нужен `num_items`.
* [guides/datasets.md](guides/datasets.md) — шардирование по записям, `drop_tail`,
  контракт паритета, кэш фильтров, HDFS.
* [guides/preprocessing.md](guides/preprocessing.md) — два бэкенда, артефакт и
  почему его надо хранить вместе с весами.
* [guides/models.md](guides/models.md) — как блоки складываются в пайплайн:
  табличный и последовательностный стек, early/late fusion, uplift, multi-task.
* [guides/metrics.md](guides/metrics.md) — контракт метрики, сведение по
  рангам, композиция обёрток, как написать свою.

## Справочники

Каталоги по пакетам: что там лежит и что из этого брать.

* [reference/pipeline.md](reference/pipeline.md) — пайплайны и их блоки.
* [reference/metrics.md](reference/metrics.md) — метрики, сборщики,
  бенчмарки, обёртки.
* [reference/nn.md](reference/nn.md) — эмбеддинги, энкодеры, агрегации и что
  с чем сочетается.

## Остальные разделы

| раздел | что внутри |
|---|---|
| [data/](data/) | формат батчей событийных последовательностей, чтение parquet |
| [best_practices/](best_practices/) | приёмы, которые экономят память и время |
| [mlp_benchmark/](mlp_benchmark/) | рецепт запуска MLP-бенчмарка по кампаниям |
| [decisions/](decisions/) | проектные документы: почему код устроен так, а не иначе |

* [data/event_sequence_batch.md](data/event_sequence_batch.md) — устройство
  `EventSequenceBatch`: что лежит в полях, как устроен паддинг.
* [data/working_with_parquet.md](data/working_with_parquet.md) — низкоуровневые
  примитивы чтения parquet.
* [best_practices/gradient_checkpointing.md](best_practices/gradient_checkpointing.md)
  — как включить gradient checkpointing и чем за это платишь.
* [best_practices/performance_metrics.md](best_practices/performance_metrics.md)
  — блок `logging:`: пропускная способность, перекос между рангами, диагностика
  шардирования. Доступен только тем, кто читал исходники, — до этой страницы.
* [mlp_benchmark/instruction.md](mlp_benchmark/instruction.md) — конфигурация
  `MLPCampaignBenchmark`.

## Проектные документы

`decisions/` — записи о принятых решениях: постановка задачи, рассмотренные
варианты, что в итоге сделано. Они описывают **прошлые состояния кода** и
намеренно упоминают классы, которых уже нет, поэтому автоматические проверки
документации их не покрывают.

| документ | о чём |
|---|---|
| [decisions/data_redesign.md](decisions/data_redesign.md) | переработка `avatar/data`: шардирование по записям, HDFS, кэш фильтров |
| [decisions/train_redesign.md](decisions/train_redesign.md) | отказ от `accelerate`, колбэки, вынесение лосса в отдельный модуль |
| [decisions/tabular_refactor.md](decisions/tabular_refactor.md) | раскладка `avatar/nn/tabular`, `STEv2` → `TabularTransformer` |
| [decisions/sequential_refactor.md](decisions/sequential_refactor.md) | раскладка `avatar/nn/sequential` |
| [decisions/documentation.md](decisions/documentation.md) | план документирования репозитория (этот раздел — его результат) |
| [decisions/maintenance.md](decisions/maintenance.md) | чистка репозитория и рефакторинг `avatar/nn/embedding` |

## Как документация не протухает

В `tests/docs/` лежат три проверки, которые запускаются вместе с обычными
тестами:

* `tests/docs/test_doc_references.py` — каждое имя вида `avatar.*`, упомянутое
  в markdown, должно импортироваться;
* `tests/docs/test_config_targets.py` — каждый `_target_` в каждом YAML должен
  разрешаться в существующий класс;
* `tests/docs/test_referenced_paths.py` — каждый файл, на который ссылается
  документация, должен существовать.

Переименование класса ломает эти проверки сразу, а не через полгода, когда на
битую ссылку наткнётся читатель. Если вы переносите или переименовываете
что-то — запустите `python -m pytest tests/docs` и почините то, что покраснело.
