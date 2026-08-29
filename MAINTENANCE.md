# Maintenance / tech-debt backlog

Составлено 2026-08-29 по результатам аудита репозитория. Задачи сгруппированы по
приоритету. Ничего из этого не меняет поведение обучения/инференса — это гигиена
упаковки, тестов и репозитория.

---

## P0 — сломано, блокирует прогон тестов

### 1. `tests/data/test_fixed_horizon_dataset.py` не собирается
`ImportError: cannot import name 'FixedHorizonDataset' from 'avatar.data.dataset'`.
Класса `FixedHorizonDataset` больше нет нигде в коде (есть `FixedHorizonCollateFn`
и `generate_fixed_horizon_dataset`, но не датасет). Тест ссылается на удалённый API.
→ Переписать на актуальный `EventSequenceDataset` + `FixedHorizonCollateFn` **или**
удалить файл (2 теста). Из-за ошибки сборки падает весь `pytest` без `--ignore`.

**Комментарий:** удаляем все, что связанно с FixedHorizonDataset, hotpp benchmark из пайплайна

### 2. `tests/data/collate_fn/test_tabular_collate_fn.py::test_tabular_collate_fn_hidden_state` — FAILED
`AttributeError: 'Tensor' object has no attribute 'keys'` в
`avatar/data/dataset/collate_fn/tabular_collate_fn.py:65`. Код ожидает, что
`tab_features[0]["_hidden_states"]` — это `dict`, тест передаёт `Tensor`.
→ Определить, кто прав (контракт `_hidden_states`), починить код или фикстуру. 

**Комментарий:** действительно в моделях сделали поддержку мн-ва эмбеддингов за раз, поэтому надо подправить тест

### 3. `tests/spark/*` падают на дефолтном JDK, нет skip-guard
5 файлов создают `SparkSession` на уровне фикстур без проверки Java. На этой машине
дефолт — JDK 25, на котором Spark 3.5 не стартует (`getSubject is not supported`).
Только `tests/local/conftest.py` умеет находить Temurin 17.
→ Вынести логику подбора JDK (8/11/17) из `tests/local/conftest.py` в общий
`tests/conftest.py`; в `tests/spark` добавить `pytest.importorskip` / skip при
отсутствии рабочего JDK.

---

## P1 — упаковка проекта

### 4. Консолидировать метаданные в `pyproject.toml`
Сейчас: `setup.py` (legacy, `install_requires=[]`, пустой `description`, автор
`NAnRusakov@sberbank.ru`, `python_requires>=3.8` при реальном 3.10) + `pyproject.toml`
только с конфигом `ruff` — нет ни `[build-system]`, ни `[project]`.
→ Добавить `[build-system]` (setuptools) и `[project]` (name, version, authors,
`requires-python = ">=3.10"`, `dependencies`, `optional-dependencies`).
`setup.py` удалить или свести к `from setuptools import setup; setup()`.

### 5. Разложить зависимости по extras
`requirements.txt` смешивает всё в кучу. Предлагаемое деление:
- core: `torch`, `numpy<2`, `pandas>=2`, `pyarrow>=14`, `hydra-core`, `omegaconf`,
  `transformers`, `accelerate`, `mlflow`, `xxhash`
- `[spark]`: `pyspark==3.5.0` (нужен для `avatar.preprocessing.spark`, `avatar.synth`)
- `[uplift]`: `scikit-uplift`, `betacal`, `catboost`
- `[dev]`: `pytest`, `pytest-cov`, `pre-commit`, `ruff`, `torchinfo`
Отдельные проблемы:
- `pynvml==13.0.1` — deprecated, torch ругается → `nvidia-ml-py`.
- `polars==0.20.10` — используется только в `avatar/metrics/campaign.py` (6 мест).
  Либо обновить пин и задокументировать зачем, либо переписать на pandas/pyarrow и убрать. 
  **Комментарий:** Давай метрику перепишем с поларса, на пандас

- Непоследовательные пины: почти всё `==`, а `pyarrow`/`pandas` — `>=`. Привести к одному стилю.
- Блок комментариев про ручную установку `hotpp-benchmark` / `torch-linear-assignment`
  перенести в `docs/` или README, из `requirements.txt` убрать.
  **Комментарий:** удаляем все связанное с hotpp & torch linear assignment

### 6. Единый источник версии
`setup.py` → `0.0.2`; в `avatar/__init__.py` нет `__version__`.
→ Определить `__version__` в пакете, в `[project]` брать `dynamic = ["version"]`.

### 7. Опечатка в имени модуля `avatar/training_agruments.py`
`agruments` → `arguments`. Импортируется в `avatar/__init__.py`.
→ Переименовать, оставить `training_agruments.py` как shim с
`from .training_arguments import *` + `DeprecationWarning` на один релиз.
---

## P2 — гигиена тестов

### 8. Добавить `[tool.pytest.ini_options]` в `pyproject.toml`
Сейчас нет конфигурации pytest вообще. Нужно:
- `testpaths = ["tests"]`
- регистрация маркеров: `hotpp`, `slow` (сейчас `PytestUnknownMarkWarning`)
- `addopts = "-m 'not slow and not hotpp'"` по умолчанию (полный прогон сейчас
  > 2 мин из-за неотфильтрованных `@pytest.mark.slow`)
- `filterwarnings` для шумных deprecation'ов (pynvml, PYARROW_IGNORE_TIMEZONE)

### 9. `passing_tests.txt` в корне
Курируемый список «зелёных» тестов, ни на что не ссылается, уже устарел (в нём нет
`tests/local/*`, часть путей не совпадает). → Либо удалить, либо оформить как
осознанный allowlist для CI с комментарием, почему остальные исключены.

### 10. `tests/metrics/test_t_map.py` — весь под `@pytest.mark.hotpp`
Требует корпоративный пакет `hotpp`. → `pytest.importorskip("hotpp")` в модуле,
чтобы прогон был зелёным без него.
**Комментарий:** удаляем эту метрику и эти тесты

### 11. Нет `tests/conftest.py` / `tests/__init__.py`
Общие фикстуры (JDK, генерация parquet, spark_session) дублируются между
`tests/local` и `tests/spark`. → Общий корневой conftest.

### 12. Задокументировать матрицу прогонов
`tests/local/` parity-тесты авто-скипаются без JDK. Явно описать в README тестов:
что гоняется всегда, что требует `[spark]` + JDK 8/11/17, как выставить `JAVA_HOME`.

---

## P3 — чистка репозитория

### 13. Незакоммиченные удаления в рабочем дереве
`git status`: удалены (но остаются в git) `preprocessing.md` и
`examples/AB/new_readme.md`. → Решить: восстановить или `git rm` + коммит.
`preprocessing.md` — это дизайн-док не-Spark бэкенда, на него ссылаются
`examples/tabular_preprocessing/README.md` и `examples/eventsequence_preprocessing/README.md`
(`../../preprocessing.md`). Скорее всего восстановить (или перенести в `docs/`
и поправить ссылки).

### 14. Битые ссылки в примерах
После удаления `preprocessing.md` ломается `[../../preprocessing.md]` в обоих
README `examples/*_preprocessing/`. Синхронизировать с решением по п.13.

### 15. ~30 каталогов `.ipynb_checkpoints/` на диске
Часть — внутри пакета `avatar/` со stale `*-checkpoint.py`
(напр. `avatar/preprocessing/spark/pipeline/.ipynb_checkpoints/sequence_preprocessor-checkpoint.py`,
`avatar/nn/.ipynb_checkpoints/`, `avatar/pipeline/**/.ipynb_checkpoints/`). Уже в
`.gitignore`, не трекаются, но мешают grep/навигации и путают инструменты.
→ `find . -name .ipynb_checkpoints -not -path './.venv/*' -exec rm -rf {} +`.

### 16. Нет корневого `README.md`
→ Добавить: что такое avatar_fm, установка (`pip install -e .[spark,dev]`),
быстрый старт, как гонять тесты, ссылки на `docs/` и `examples/`.

### 17. `examples/README.md` устарел
В «TODO» указан «Preprocessing Sequence Data» — уже сделан. В «Навигации» нет
`tabular_preprocessing/` и `eventsequence_preprocessing/`. Контакт —
`nanrusakov@sberbank.ru` (тот же домен, что в `setup.py`; сверить актуальность).

### 18. `.gitignore` — стоковый шаблон 190 строк
- `avatar_fuxictr/**/*.parquet` — каталога `avatar_fuxictr` не существует, удалить.
- Бланкетные `**/*.parquet`, `**/*.csv`, `**/*.pkl` могут скрыть нужные фикстуры
  в `tests/data/` — сузить до `**/data/`, `examples/**/data/`, `experiments/**/data/`.
- Убрать неактуальные секции (Django/Flask/Scrapy/PyBuilder/SageMath).

### 19. `.pre-commit-config.yaml` отсутствует
`pre-commit==4.2.0` в зависимостях, конфига нет. → Добавить (`ruff`, `ruff-format`,
`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`)
или убрать зависимость.

### 20. Конфиг `ruff` в `pyproject.toml`
`preview = true`, выбраны только `E/F/I/W`. → Рассмотреть `B` (bugbear), `UP`
(pyupgrade), `RUF`. Добавить `[tool.ruff.format]`, раз есть `pre-commit`.
Прогнать `ruff check .` и завести отдельную задачу под накопившиеся замечания.

---

## P4 — структурное (крупнее, опционально)

### 21. Ленивые импорты тяжёлых бэкендов
`avatar/synth/generate_synth.py` и `avatar/metrics/campaign.py` импортируют
`pyspark` / `polars` на уровне модуля → это делает их жёсткой зависимостью для
`import avatar.synth` / `avatar.metrics`. Сделать как в
`avatar/preprocessing/__init__.py` (lazy `__getattr__`).

**Комментарий:** модуль avatar/synth полностью удаляем из репозитория

### 22. Симметрия публичного API препроцессинга
`avatar.preprocessing.local` экспортирует `LabelEncoder`/`StandardScaler` на верхнем
уровне, а `spark` — только через `spark.label_encoder` / `spark.standard_scaler`.
Привести к единому виду.

### 23. Крупные модули
`avatar/data/dataset/tabular_dataset.py` (48K), `avatar/train.py` (36K),
`avatar/metrics/supervised.py` (36K), `avatar/metrics/horizon_metric.py` (36K) —
кандидаты на разбиение. Делать только точечно и под отдельную задачу.

**Комментарий:** avatar/metrics/horizon_metrics.py - удаляем
