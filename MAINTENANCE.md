# Maintenance / tech-debt backlog

Аудит от 2026-08-29. **Все пункты выполнены** в этой сессии (ветка `master`,
локальные коммиты `5432b95..HEAD`). Ниже — что и как сделано.

---

## P0 — сломано, блокировало прогон тестов

### 1. `tests/data/test_fixed_horizon_dataset.py` не собирался ✅
Удалено всё, что связано с `FixedHorizonDataset` и hotpp: файл теста удалён,
`avatar/synth/generate_synth_fixed_horizon.py` удалён вместе со всем
`avatar/synth/` (см. B/№21).

### 2. `test_tabular_collate_fn_hidden_state` — FAILED ✅
Контракт `_hidden_states` стал `dict[str, Tensor]` (несколько внешних эмбеддингов
за раз). Фикстура теста передавала голый тензор → правка теста: ключ `ext_emb`,
проверки по ключу. Аналогично починены `test_process_tabular_hidden_state` и
`test_attention_encoder` (см. F4).

### 3. `tests/spark/*` без skip-guard по JDK ✅
Логика подбора JDK 8/11/17 вынесена в общий `tests/conftest.py`
(`_ensure_java` + `spark_session`/`spark` фикстуры со `skip`, если JDK нет).
Инлайновые `SparkSession` в 5 файлах `tests/spark/` удалены.

---

## P1 — упаковка проекта

### 4. Метаданные в `pyproject.toml` ✅
Добавлены `[build-system]` (setuptools) и `[project]` (deps, extras
`spark`/`catboost`/`dev`, `requires-python = ">=3.10"`, dynamic version).
`setup.py` удалён.

### 5. Зависимости по extras ✅
`requirements.txt` → тонкая обёртка `-e .[spark,catboost,dev]`. `pynvml` →
`nvidia-ml-py`. `polars` удалён (метрика переписана на pandas, см. D1).
`hotpp-benchmark` / `torch-linear-assignment` / `pytorch-lifestream` (не
использовался нигде) убраны.

### 6. Единый источник версии ✅
`avatar/_version.py` (`__version__`), pyproject `dynamic = ["version"]` через
`[tool.setuptools.dynamic]`. Версия поднята `0.0.2 → 0.1.0`.

### 7. Опечатка `training_agruments.py` ✅
Переименован в `training_arguments.py`; старое имя оставлено как shim с
`DeprecationWarning`. Импорты (`avatar/__init__.py`, `avatar/train.py`) обновлены.

---

## P2 — гигиена тестов

### 8. `[tool.pytest.ini_options]` ✅
`testpaths`, маркер `slow` зарегистрирован, `addopts = "-m 'not slow' -ra"`,
`filterwarnings` для pynvml / PYARROW_IGNORE_TIMEZONE / LooseVersion.

### 9. `passing_tests.txt` ✅ удалён.

### 10. `tests/metrics/test_t_map.py` + t_map метрика ✅
Удалены вместе с `avatar/metrics/horizon_metric.py` (`TMAPMetric` из hotpp).

### 11. `tests/conftest.py` ✅ создан (общие фикстуры, синтетика без Spark).

### 12. Матрица прогонов ✅ описана в `tests/README.md` и `README.md`.

---

## P3 — чистка репозитория

### 13–14. Незакоммиченные удаления + битые ссылки ✅
`preprocessing.md` и `examples/AB/new_readme.md` удалены (`git rm`); ссылки
`../../preprocessing.md` вырезаны из обоих `examples/*_preprocessing/README.md`.

### 15. `.ipynb_checkpoints/` ✅ ~30 каталогов удалено; правило в `.gitignore`.

### 16. Корневой `README.md` ✅ добавлен (что это, установка, layout, тесты/JDK).

### 17. `examples/README.md` ✅ обновлён (новые примеры, снят done-TODO, убран
контакт).

### 18. `.gitignore` ✅ переписан: убран несуществующий `avatar_fuxictr`,
бланкетные `**/*.parquet|csv|pkl` сужены до data-каталогов, выкинуты стоковые
Django/Flask/Scrapy/PyBuilder/SageMath секции.

### 19. `.pre-commit-config.yaml` ✅ добавлен (ruff + ruff-format + hygiene hooks).

### 20. Конфиг `ruff` + полная зачистка ✅
Добавлены `B`/`UP`/`RUF` + `[tool.ruff.format]`. `ruff check .` и
`ruff format --check .` — чисто (было 497 замечаний, 111 файлов
переформатировано). B008 (`nn.*Loss()` в дефолтах) — рефактор на `None` +
инстанс в теле.

---

## P4 — структурное

### 21. `avatar/synth/` ✅ удалён полностью; тестовая синтетика перенесена в
`tests/conftest.py` (pandas/pyarrow, без Spark). `avatar/metrics/campaign.py`
переписан с polars на pandas (D1) — единственный оставшийся тяжёлый
модуль-левел импорт вне preprocessing.

### 22. Симметрия API препроцессинга ✅
`avatar.preprocessing.spark` теперь ре-экспортирует
`LabelEncoder`/`StandardScaler`/`NumCatPipeline`/`TabularPreprocessor`/
`EventSequencePreprocessor` с верхнего уровня, как `local`.

### 23. Крупные модули / `horizon_metric.py` ✅
`avatar/metrics/horizon_metric.py` (36K) удалён. Разбиение
`tabular_dataset.py` / `train.py` — не делалось (отдельная задача, риск).

---

## Итог

* Дефолтный прогон: **144 passed**, 66 slow — тоже passed, 0 упавших.
  Spark-тесты (55) и local-тесты (31) зелёные с JDK 17, скипаются без него.
* Оба примера (`examples/*_preprocessing/run_both_backends.py`) проходят
  end-to-end с кросс-загрузкой артефактов.
* `ruff check .` / `ruff format --check .` — чисто.
* `pip install -e .` работает, версия читается из `avatar/_version.py`.

### Открытые вопросы для владельца

* Версия поднята до `0.1.0` — при желании откатить в `avatar/_version.py`.
* `avatar/data/dataset/tabular_dataset.py` (48K) и `avatar/train.py` (36K) —
  кандидаты на разбиение, оставлены как есть.
* Extras `spark`/`catboost` объявлены, но `avatar.metrics` всё ещё жёстко тянет
  `scikit-uplift` / `betacal` (модуль-левел импорт в `metrics/uplift.py`).
