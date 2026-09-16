# Окружение для локальных AutoML-тестов

Эта инструкция относится только к проверочным ноутбукам из
`examples/automl/tests`. Общая установка репозитория — в корневом `README.md`.

Поддерживаемое окружение: Linux, Python 3.11 или 3.12.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[boosting,dev]"
```

`environment.venv_path` для Osiris должен указывать на абсолютный POSIX-путь к
shared окружению на кластере. Job запускается через `<venv_path>/bin/python`,
без `PYTHONPATH`.

При первом local-вызове AutoML процесс устанавливает `CUDA_VISIBLE_DEVICES=0`
и не восстанавливает предыдущее значение. Для однодевайсных
boosting/inference job на Osiris также используется device 0.

Parity-ноутбуки против `autocampaignxfm` и каталог `tools/automl_parity` при
переносе в avatar не копировались: они рассчитаны на конфигурацию до удаления
`model_scope`/`report_month_column` и на пакет, которого нет ни в одном из
репозиториев. См. [docs/decisions/automl_migration.md](../../../docs/decisions/automl_migration.md), решение D9.

## Checkpoint реестра метрик

`design_fix_quality.ipynb` запускает только checkpoint нового metric API. Он не генерирует данные и не создаёт новый baseline: используются фиксированные пять datasets по 200 000 строк из `<root>/data`, их `<root>/dataset_fingerprints.json` и reference `checkpoints/after_point_8`.

Откройте notebook из этого каталога в editable окружении `avatar` и выполните все code cells один раз. Он создаст отдельный `checkpoints/after_metric_registry`, передаст task-specific `optimization_metric` и полный явный стандартный список в `evaluate(metrics=...)`, затем проверит с порогом `1e-8` порядок identity/model-branch keys, все scores и evaluation/validation metrics, а также exact best params и semantic config. Итоговый `<root>/summary.json` содержит фактические дельты, dataset/config fingerprints, commit/dirty-state, длительности и пути task artifacts.

Эквивалентный запуск без Jupyter из `examples/automl/tests`:

```bash
../../../.venv/bin/python design_fix_quality.py checkpoint \
  --root ../../../outputs/design_fix_migration \
  --name after_metric_registry
```

Имя checkpoint должно быть новым: существующий каталог намеренно не перезаписывается.
