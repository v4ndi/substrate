# Окружение для локальных AutoML-тестов

Эта инструкция относится только к проверочным ноутбукам и parity-скриптам из
`examples/automl/tests`. Она не является общей инструкцией установки fmlib.

Поддерживаемое окружение: Linux, Python 3.11 или 3.12. Если базовый `env` fmlib
уже собран, активируйте его и доустановите зависимости бустингов:

```bash
source env/bin/activate
python -m pip install \
  "betacal>=1.1,<2" \
  "catboost>=1.2.9,<2" \
  "matplotlib>=3.8,<4" \
  "optuna>=4,<5" \
  "xgboost>=2.1,<4" \
  "XlsxWriter>=3.2,<4"
```

Для установки всего проекта вместе с инструментами тестирования из корня
репозитория:

```bash
python -m pip install -e ".[dev]"
```

Для конфигов и parity-тестов задайте абсолютные корни:

```bash
export SBER_INFRA_ROOT=/absolute/path/to/sber_infra
export FMLIB_ROOT=/absolute/path/to/sber_infra/fmlib-main
```

`environment.venv_path` для `batch` и `supercomp` должен указывать на
абсолютный POSIX-путь к этому же окружению. Job запускается через
`<venv_path>/bin/python`, без `PYTHONPATH`.

При первом local-вызове AutoML процесс устанавливает
`CUDA_VISIBLE_DEVICES=0` и не восстанавливает предыдущее значение. Для
однодевайсных boosting/inference job на supercomp также используется device 0.

## Checkpoint реестра метрик

`design_fix_quality.ipynb` запускает только checkpoint нового metric API. Он не генерирует данные и не создаёт новый baseline: используются фиксированные пять datasets по 200 000 строк из `outputs/design_fix_migration/data`, их `dataset_fingerprints.json` и reference `checkpoints/after_point_8`.

Откройте notebook из этого каталога в editable окружении `fmlib` и выполните все code cells один раз. Он создаст отдельный `checkpoints/after_metric_registry`, передаст task-specific `optimization_metric` и полный явный стандартный список в `evaluate(metrics=...)`, затем проверит с порогом `1e-8` порядок identity/model-branch keys, все scores и evaluation/validation metrics, а также exact best params и semantic config. Итоговый `summary.json` содержит фактические дельты, dataset/config fingerprints, commit/dirty-state, длительности и пути task artifacts.

Эквивалентный запуск без Jupyter из `fmlib-main/examples/automl/tests`:

```bash
../../../env/bin/python design_fix_quality.py checkpoint \
  --root ../../../../outputs/design_fix_migration \
  --name after_metric_registry
```

Имя checkpoint должно быть новым: существующий каталог намеренно не перезаписывается.
