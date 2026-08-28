# Design: non-Spark реализация `avatar.preprocessing`

Статус: draft, обсуждение. Реализацию не начинать до утверждения.
Автор контекста: сессия 2026-08-29. Связано с `TODO.md` (миграция препроцессинга
на torch/GPU) и `[[avatar-fm-tabular-preproc]]`.

---

## 1. Цель

Сейчас `avatar.preprocessing` содержит только Spark-реализацию
(`avatar/preprocessing/spark/`). Нужна вторая реализация, работающая **без Spark /
JVM / YARN**, на одной машине с ограничениями:

- **8 CPU, 128 GB RAM**
- датасеты до **100M строк × 500 признаков**
- вход/выход — parquet (как сейчас)

Требование совместимости: артефакт (`dump()` / `load()`) и выход (`cat_features` /
`num_features`, либо широкая табличка — см. `TODO.md`) должны совпадать со Spark-путём
**побитово** (в пределах fp-погрешности), чтобы:

- не было train/serving skew (ср. дисциплину bit-identical inference в `[[fmlib-mtl-pr19]]`);
- downstream (`avatar/data/dataset/tabular_dataset.py`, `TabularEmbedding`, инференс)
  не менялся;
- артефакт, обученный любым backend-ом, читался обоими.

Пользовательский API должен зеркалить Spark: `fit` / `transform` / `fit_transform` /
`dump` / `load`.

---

## 2. Анализ ограничений: почему streaming обязателен

100M × 500. Грубая оценка «в памяти» (Arrow/несжато):

- 400 числовых `float32` + 100 категориальных `int16`:
  `100e6 × (400·4 + 100·2) B ≈ 180 GB` — **не влезает в 128 GB**.
- Даже только числовой блок для расчёта статистик: `100e6 × 400 × 4 B ≈ 160 GB`.
- На диске parquet (сжатый) — порядка 40–70 GB.

Выводы:

- **Нельзя** делать `pd.read_parquet(...)` / `pl.read_parquet(...)` целиком, и даже
  `LazyFrame.collect()` без агрегации.
- И `fit`, и `transform` **обязаны быть потоковыми** (chunk-by-chunk по row-group /
  батчам фиксированного размера), с ограниченной памятью O(batch × n_cols) +
  O(размер статистик).
- `fit` — 1 потоковый проход (максимум 2, см. §7.1 про стабильность std).
- `transform` — 1 потоковый проход с инкрементальной записью parquet (`ParquetWriter`
  / `sink_parquet`), выход **никогда** не материализуется целиком.

Ориентир по времени: один проход по ~50 GB parquet на 8 ядрах — I/O-bound,
~5–20 мин.

---

## 3. Ключевая идея архитектуры

Разделить на **backend-agnostic ядро** и **backend-specific сканер данных**.

Вся «умная» часть препроцессинга — статическая и уже описывается маленьким словарём
(`TabularPreprocessor.dump()`):

- `LabelEncoder`: per-column `value -> id`, null/unseen -> `unk` (0).
- `StandardScaler`: опц. `signed_log1p` -> `(x - mean) / (std + 1e-8)` -> `fillna(0)`.
- `TabularPreprocessor`: кумулятивный `offset_map`, упаковка в массивы, `vocab_size`.
- `EventSequencePreprocessor`: + `sort(id, time)` + `date_to_scaled_unix` +
  `groupby(id).agg(collect_list)`.

Backends отличаются **только**:

1. как просканировать датасет для сбора статистик (`fit`);
2. как применить набор column-выражений и записать результат (`transform`).

Математика, схема артефакта, построение `offset_map`, `dump_tabular_meta` — общие.

---

## 4. Структура пакета

```
avatar/preprocessing/
  __init__.py            # ленивые импорты обоих backends (см. §11)
  base/                  # NEW — backend-agnostic
    __init__.py
    artifact.py          # схема dump()-словаря, валидация, yaml IO, версия/бэкенд
    accumulators.py      # потоковые аккумуляторы: MeanStdAccumulator,
                         #   ValueCountAccumulator, + слияние чанков
    encode.py            # чистая математика transform (numpy-выражения):
                         #   signed_log1p, standardize, encode_categorical, apply_offsets
    offsets.py           # build_offset_map(values_to_id, spec_tokens) -> offset_map, vocab_size
    meta.py              # dump_tabular_meta (перенести из spark/utils.py, обобщить)
    io.py                # BatchSource: итератор Arrow RecordBatch по glob/paths/Dataset
  spark/                 # БЕЗ ИЗМЕНЕНИЙ
    ...
  local/                 # NEW — однопроцессорная реализация
    __init__.py
    label_encoder.py     # LocalLabelEncoder(BasePreprocessor)
    standard_scaler.py   # LocalStandardScaler(BasePreprocessor)
    pipeline/
      __init__.py
      base_pipe.py       # LocalNumCatPipeline(BaseDataPipeline)
      tabular_pipe.py    # local TabularPreprocessor
      sequence_preprocessor.py
```

Общие ABC (`BasePreprocessor`, `BaseDataPipeline`) сейчас лежат в `spark/`, но
**не** импортируют Spark в теле — только сигнатуры. Их нужно поднять в `base/` и
оставить в `spark/` реэкспорт для обратной совместимости:

```python
# avatar/preprocessing/spark/base_preprocessor.py
from avatar.preprocessing.base import BasePreprocessor  # noqa: F401
```

---

## 5. Общий контракт артефакта

`base/artifact.py` фиксирует точную схему словаря, который производят/читают **оба**
backend-а. Совпадает с текущим `TabularPreprocessor.dump()`:

```yaml
# tabular preprocessor artifact
_version: 2                 # NEW: версия схемы
_backend: "local"           # NEW: чем обучено (информативно, load игнорирует)
cat_cols: [<str>, ...] | null
num_cols: [<str>, ...] | null
label_encoder:
  columns: [<str>, ...]
  spec_tokens: {unk: 0}
  frequency_encoder: <bool>
  values_to_id: {<col>: {unk: 0, <value>: <int>, ...}, ...}
standard_scaler:
  columns: [<str>, ...]
  fillna: <bool>
  to_log_columns: [<str>, ...]
  mean_std: {<col>: {mean: <float>, std: <float>}, ...}
spec_tokens: {pad: 0}
offset_map: {<col>: <int>, ...}
vocab_size: <int>
```

Для `EventSequencePreprocessor` — плюс `groupby_columns`, `id_column`,
`event_time_column`, `event_type_ids_columns`, `time_unit`.

Гарантии, закреплённые тестами (§10):

1. `local.TabularPreprocessor.fit(df).dump()` даёт словарь, который принимают
   `spark.TabularPreprocessor.load()` **и** `local.TabularPreprocessor.load()`.
2. При одном и том же `values_to_id` / `mean_std` `transform` обоих backend-ов
   выдаёт идентичные `cat_features` (точно) и `num_features` (|Δ| < 1e-5).
3. `_version` / `_backend` необязательны при чтении старых артефактов (default:
   `_version=1`).

`load()` не зависит от backend-а сборки: и Spark-, и local-артефакт грузится в
любой класс. Отличаться может только **пере-`fit`** (см. §7.3 про детерминизм).

---

## 6. Backend-agnostic ядро (`base/`)

### 6.1 `io.BatchSource`

```python
class BatchSource:
    def __init__(self, paths: str | list[str], columns: list[str] | None = None,
                 batch_rows: int = 250_000): ...
    def batches(self) -> Iterator[pa.RecordBatch]: ...   # pyarrow.dataset, hive-partitioning
    def num_rows(self) -> int: ...
```

Строится на `pyarrow.dataset.dataset(paths, format="parquet", partitioning="hive")`
+ `to_batches(columns=..., batch_size=...)` — ровно как уже делает
`avatar/data/parquet.py:read_parquet_file`, но без разворачивания в dict по строкам.
`batch_rows` подбирается так, чтобы `batch_rows × n_cols × 8B` ≲ 1–2 GB.

Зависимости: только `pyarrow` (уже есть) + `numpy` (уже есть). polars — опциональный
ускоритель (§7.4), не hard-dep.

### 6.2 `accumulators`

**`MeanStdAccumulator(columns, to_log_columns)`** — потоковый расчёт mean/std,
паритетный со Spark `F.mean` / `F.stddev`:

- Spark `F.stddev` = **выборочное** стд (ddof=1). Обязательно то же:
  `std = sqrt((Σx² − n·mean²) / (n − 1))`.
- null игнорируются (как в Spark-агрегатах) — вести `count` по не-null отдельно на
  колонку.
- для `to_log_columns` перед аккумуляцией применяется `signed_log1p` (Spark считает
  mean/std уже по лог-трансформированным значениям).
- слияние чанков — формулы Чана (parallel variance), в `float64`.
- **стабильность**: наивный `Σx²` рискован при больших значениях. Варианты:
  (а) центрирование по грубому среднему из первого чанка (shifted-data);
  (б) Welford/Чан по чанкам (рекомендуется, дешёво);
  (в) второй проход. По умолчанию — (б).
- edge case: `n == 1` -> Spark `F.stddev` = null -> в артефакте `std=nan`.
  Решить явно: писать `std=0.0` и логировать WARNING (иначе `(x-mean)/(nan+1e-8)`
  ломает весь столбец).

**`ValueCountAccumulator(columns, max_cardinality)`** — `dict[col] -> Counter`:

- потоковый `value -> count` на колонку; null не учитывается (пойдёт в `unk`).
- **cardinality guard**: если `len(counter) > max_cardinality` (default 1e6) —
  `raise` с понятным сообщением (для id-подобных колонок нужен хеш-эмбеддинг, не
  label encoding — это вне scope). Опция `on_overflow="topk"` -> оставить top-K по
  частоте, остальное схлопнуть в `unk`.
- память: O(Σ distinct по колонкам). Для кейса «500 признаков, категориальные
  низкокардинальные» (как в примере: 52 кат-колонки, vocab 172) — байты. Guard
  ловит патологию.

### 6.3 `offsets.build_offset_map`

Точная копия логики `TabularPreprocessor.fit`:

```python
vocab_size = len(spec_tokens) if cat_cols else 0
for col in cat_cols:                       # порядок = порядок cat_cols
    offset_map[col] = vocab_size
    vocab_size += len(values_to_id[col])   # включая unk
```

### 6.4 `encode` — чистая математика (numpy, на один батч)

```python
def signed_log1p(x):                        # паритет со spark.standard_scaler.signed_log1p
    return np.log1p(np.abs(x)) * np.sign(x) # np.sign(0)==0 -> совпадает с when(x==0,0)

def standardize(x, mean, std, fillna):      # x: (n, n_num) float64
    z = (x - mean) / (std + 1e-8)
    if fillna: z = np.where(np.isnan(z), 0.0, z)   # ВНИМАНИЕ: Spark fillna чинит
    return z                                        # null ДО NaN; см. §12

def encode_categorical(codes_2d, per_col_mapping, unk_id=0):
    # codes_2d: (n, n_cat) исходные значения; на выход — локальные id, null/unseen -> unk
    ...

def apply_offsets(local_ids_2d, offsets_vec):
    return local_ids_2d + offsets_vec
```

Категориальный маппинг на батче: для целочисленных кодов (типичный случай) —
`np.searchsorted` по отсортированному массиву известных значений + маска
`known` -> `where(known, mapped_id, unk)`. Для строк — `dict.get` через
`np.vectorize` или pandas `.map` (медленнее; см. §12 — строки лучше факторизовать
выше по пайплайну).

### 6.5 `meta.dump_tabular_meta`

Перенести из `spark/utils.py` (он не использует Spark по сути — только читает
атрибуты препроцессора). Оставить в `spark/utils.py` реэкспорт.

---

## 7. `local` backend

### 7.1 `fit` (потоковый, 1 проход)

```python
def fit(self, source: str | list[str] | pa.dataset.Dataset):
    src = BatchSource(source, columns=self.cat_cols_or_empty + self.num_cols_or_empty)
    mean_std = MeanStdAccumulator(self.num_cols, self.scaler.to_log_columns)
    vcounts  = ValueCountAccumulator(self.cat_cols, self.max_cardinality)
    for batch in src.batches():
        if self.num_cols: mean_std.update(batch)
        if self.cat_cols: vcounts.update(batch)
    self.scaler.mean_std   = mean_std.finalize()          # {col: {mean, std}}
    self.label_encoder.values_to_id = vcounts.finalize(   # {col: {unk:0, v:1, ...}}
        frequency_encoder=self.label_encoder.frequency_encoder,
        order=self.categorical_order,                     # см. §7.3
    )
    self.offset_map, self.vocab_size = build_offset_map(
        self.cat_cols, self.label_encoder.values_to_id, self.spec_tokens)
```

- один проход, память ограничена;
- CPU: pyarrow-декодирование parquet многопоточно; аккумуляторы — векторный numpy;
- при желании — распараллелить по файлам через `concurrent.futures.ThreadPoolExecutor`
  (GIL отпускается в pyarrow/numpy), сливать частичные аккумуляторы.

### 7.2 `transform` (потоковый, инкрементальная запись)

```python
def transform(self, source, output_path, identity_cols=None, output="packed"):
    src = BatchSource(source)                 # читаем ВСЕ колонки (passthrough)
    writer = None
    for batch in src.batches():
        out = self._transform_batch(batch, identity_cols, output)   # pa.RecordBatch
        if writer is None:
            writer = pq.ParquetWriter(output_path, out.schema)
        writer.write_batch(out)
    writer.close()
```

`_transform_batch`:

1. категориальные: `encode_categorical` -> `apply_offsets`;
2. числовые: `signed_log1p` (для `to_log_columns`) -> `standardize`;
3. `output`:
   - `"packed"` — собрать `cat_features: list<int64>`, `num_features: list<float32>`
     (Spark-совместимо, downstream не меняется);
   - `"wide"` — вернуть широкую табличку с трансформированными колонками (для
     направления «препроцессинг на GPU» из `TODO.md`);
4. `identity_cols` — приложить исходные (нетрансформированные) копии;
5. passthrough прочих колонок (target, id, report_month, hidden_state, ...) — без
   изменений.

Полностью потоковая map-операция, память ~O(batch). Выход можно партиционировать
(`pq.write_to_dataset` / вручную по файлам) для последующего `TabularDataset`.

### 7.3 Детерминизм порядка категорий (отличие от Spark)

Spark `LabelEncoder` по умолчанию использует `F.collect_set` — **порядок
недетерминирован** между запусками. Local backend назначает `id` **детерминированно**:

- `categorical_order="sorted"` (default) — по значению;
- `categorical_order="count_desc"` — по убыванию частоты (тогда совпадает с
  `frequency_encoder=True`, который в Spark и так делает `row_number` по count desc);
- `categorical_order="first_seen"` — по первому появлению в потоке.

Следствие: **пере-`fit`** Spark-ом и local-ом на одних данных может дать разные
`value -> id` (но одинаковые по смыслу). Это **не** проблема совместимости
артефакта: `id` хранятся в `values_to_id`, `load()` берёт их как есть. Проблема
только если кто-то полагается на конкретные числовые `id` от Spark-`collect_set` —
чего делать нельзя (порядок и так случайный). Задокументировать в docstring и в
release notes. Local-детерминизм — это улучшение.

### 7.4 Опциональный polars fast-path

`requirements.txt` уже содержит `polars==0.20.10`. Это **старый** streaming-движок
(ограниченный). Варианты:

- **A (рекомендуется):** reference-реализация на `pyarrow` + `numpy` (§6.1–7.2) —
  без зависимости от зрелости polars-streaming, полный контроль над памятью.
  polars используется опционально внутри `_transform_batch` для удобства выражений,
  либо вообще не используется.
- **B:** bump `polars>=1.12` и для `transform` использовать
  `pl.scan_parquet(src).with_columns([...]).sink_parquet(dst)` (новый streaming
  engine, map-операции идут потоково надёжно). Для `fit` —
  `pl.scan_parquet(src).select([...aggs...]).collect(engine="streaming")`.
  Меньше кода, но зависит от версии polars и её OOM-поведения на group_by.
- **C:** pandas — только как chunked-фолбэк (`pyarrow` `iter_batches` -> `to_pandas`
  по чанку) для малых данных / окружений без polars. Не основной путь.

Дизайн ядра (§6) от выбора не зависит: аккумуляторы и `encode` работают с numpy;
polars/pandas — лишь способ дотащить батч до numpy и записать результат.

### 7.5 `EventSequencePreprocessor` (local) — сложный кейс

`transform` требует `sort(id, time)` + `groupby(id).agg(collect_list по всем
колонкам)`. На 100M событий это самая тяжёлая операция.

Замечание: 500 признаков — это про **табличный** кейс. В последовательностях
атрибутов на событие обычно десятки, а не сотни. Оценивать нужно по реальному
числу sequence-колонок.

Стратегии (по возрастанию сложности):

1. **Требовать вход, партиционированный по `id`** (upstream пишет parquet
   bucket-ом по `hash(id) % K`). Тогда каждый бакет влезает в RAM: читаем бакет ->
   `sort` -> `groupby` -> дописываем в выход. Память O(размер бакета). **Рекомендуется
   как основной путь** — надёжно и просто.
2. **polars streaming** (`scan -> sort -> group_by -> agg -> sink`) на polars>=1.x.
   Может хватить, если sequence-колонок немного. Пробовать после (1) как удобство.
3. **Внешняя сортировка вручную:** проход 1 — раскидать строки по K временным
   файлам по `id % K`; проход 2 — обработать каждый temp-файл в памяти. K подобрать
   под 128 GB. Фолбэк, если (1) недоступно.

`date_to_scaled_unix`: Spark `F.unix_timestamp` использует **session timezone**.
В local — приводить время к UTC явно (`pa`/`pl` epoch seconds) и делить на тот же
`conversion_factors[time_unit]`. Задокументировать возможный сдвиг на TZ-offset,
если исходные Spark-данные считались в локальной TZ (§12).

---

## 8. API (зеркалит Spark)

```python
from avatar.preprocessing.local import TabularPreprocessor

pp = TabularPreprocessor(
    categorical_columns=cat_cols,
    numeric_columns=num_cols,
    spec_tokens={"pad": 0},
    standard_scaler_kwargs={"to_log_columns": log_cols, "fillna": True},
    label_encoder_kwargs={"frequency_encoder": False},
    max_cardinality=1_000_000,          # NEW
    categorical_order="sorted",         # NEW
)

pp.fit("/data/train")                                    # потоковый проход
pp.transform("/data/train", "/data/train_processed", identity_cols=["target_attr_2"])
pp.transform("/data/valid", "/data/valid_processed")
pp.transform("/data/test",  "/data/test_processed")

cfg = pp.dump()
yaml.safe_dump(cfg, open("artifacts/td_tabular_preprocessor.yaml", "w"))

# инференс / другой backend
pp2 = TabularPreprocessor.load(cfg)          # тот же словарь, что у Spark
```

Отличия сигнатур от Spark:

- `fit(source)` принимает path / glob / list / `pa.dataset.Dataset` вместо
  Spark `DataFrame`;
- `transform(source, output_path, ...)` **пишет parquet** (не может вернуть
  «ленивый» результат, т.к. out-of-core). Для `output="wide"` и малых данных можно
  дополнительно вернуть `pa.Table`.
- новые kwargs: `max_cardinality`, `categorical_order`, `output`, `batch_rows`.

---

## 9. Что НЕ меняется

- `avatar/preprocessing/spark/**` — не трогаем.
- Downstream: `TabularDataset`, `TabularCollateFn`, `TabularBatch`,
  `TabularEmbedding` — при `output="packed"` вообще не в курсе смены backend-а.
- Формат артефакта — расширяется двумя необязательными полями, старые читаются.

---

## 10. Тесты паритета

`tests/local/` (без Spark-сессии -> быстрый CI):

1. **Golden vs Spark.** Мелкий фикстур-фрейм (как `tests/spark/test_tabular_preprocessor.py`):
   прогнать `spark.TabularPreprocessor` и `local.TabularPreprocessor`, сравнить:
   - `dump()` словари после нормализации порядка категорий (сортировать
     `values_to_id` по значению перед сравнением, т.к. Spark-порядок случаен);
     `mean_std` — `pytest.approx(rel=1e-9)`;
   - трансформированные `cat_features` — точное равенство;
   - `num_features` — `np.allclose(atol=1e-5)`.
2. **Cross-load.** `local.fit().dump()` -> `spark.load()` -> `transform` == 
   `local.transform`; и наоборот.
3. **Streaming-инвариант.** Один и тот же вход, `batch_rows ∈ {1e3, 1e5, всё}` —
   идентичный артефакт и выход (проверяет корректность слияния аккумуляторов).
4. **Edge cases:** пустые категории, колонка с 1 уникальным значением, `n==1` для
   std, все-null числовая колонка, unseen категория в `transform`, null в
   категориальной (-> unk), `to_log_columns` с отрицательными/нулевыми значениями.
5. **Cardinality guard** срабатывает и `on_overflow="topk"` работает.
6. Перенести существующие `tests/spark/test_{label_encoder,standard_scaler,
   num_cat_preprocessor,tabular_preprocessor,sequence_preprocessor}.py` в
   параметризованные по backend, где возможно.

Хелпер `avatar.preprocessing.base.testing.assert_artifacts_equivalent(a, b)`.

---

## 11. Зависимости и упаковка

- Hard deps новой части: `pyarrow`, `numpy` — **уже есть**.
- `polars` — уже в `requirements.txt` (`0.20.10`). Решить: bump до `>=1.12` для
  пути B (§7.4) или оставить и идти путём A. Рекомендация: **путь A**, polars bump
  отдельным PR если понадобится.
- `avatar/preprocessing/__init__.py` — ленивые импорты, чтобы ни `pyspark`, ни
  `polars` не тянулись на `import avatar.preprocessing`:

```python
def __getattr__(name):
    if name == "spark":
        from . import spark as _m; return _m
    if name == "local":
        from . import local as _m; return _m
    raise AttributeError(name)
```

(сейчас `__init__.py` делает безусловный `from . import spark` — это заставляет
иметь pyspark даже для local-пути; исправить.)

---

## 12. Риски и открытые вопросы

| # | Риск | Митигация |
|---|------|-----------|
| 1 | `F.stddev` = выборочное (ddof=1); легко случайно сделать ddof=0 | зафиксировать в `MeanStdAccumulator`, golden-тест |
| 2 | Наивный `Σx²` теряет точность на больших значениях | Welford/Чан по чанкам в float64 (§6.2) |
| 3 | `n==1` -> Spark `std=null`/nan -> ломает столбец | писать `std=0.0` + WARNING |
| 4 | Spark `fillna` в scaler чинит **null**, а не NaN; `signed_log1p` от NaN = NaN; `(x-mean)/std` от NaN = NaN, и `fillna` его НЕ уберёт | воспроизвести Spark 1-в-1: `fillna` до масштабирования? Нет — Spark делает `select(scale)` затем `fillna(0.0)` по null. NaN на входе -> NaN на выходе в обоих. Тестом зафиксировать текущее поведение, не «улучшать» |
| 5 | `collect_set` порядок случаен -> разные id при пере-fit | local детерминирован (§7.3), документировать; артефактная совместимость не страдает |
| 6 | Высококардинальные категориальные -> OOM в `ValueCountAccumulator` | cardinality guard + `on_overflow="topk"` |
| 7 | Строковые категории -> медленный map на батче | рекомендовать факторизацию выше по пайплайну (int-коды); поддержать строки, но не оптимизировать |
| 8 | `date_to_scaled_unix`: Spark session TZ vs UTC | явный UTC в local, документировать возможный сдвиг; добавить `tz` параметр |
| 9 | Sequence `groupby` на 100M не влезает | требовать id-партиционированный вход (§7.5, стратегия 1) |
| 10 | polars 0.20.10 streaming слабый | путь A на pyarrow/numpy (§7.4) |
| 11 | Порядок колонок в `cat_features`/`num_features` = порядок `cat_cols`/`num_cols` | брать строго из артефакта, тест на стабильность layout |
| 12 | `identity_cols` с именем-коллизией `source_<col>` | как в Spark — задокументировать ограничение |

---

## 13. План внедрения (фазы)

1. **Ядро.** `base/` (artifact, accumulators, encode, offsets, io, meta) +
   поднять ABC из `spark/` с реэкспортом. Ленивые импорты в `__init__.py`.
   Юнит-тесты на аккумуляторы и `encode` (без данных).
2. **`local.TabularPreprocessor`** — `fit` / `transform` (`packed` + `wide`) /
   `dump` / `load`. Тесты паритета §10 (1–5).
3. **`local` LabelEncoder / StandardScaler / NumCatPipeline** как самостоятельные
   классы (для тех, кто использует их отдельно) + перенос `tests/spark/*` в
   параметризованные.
4. **`local.EventSequencePreprocessor`** — стратегия 1 (id-партиции) + фолбэк 3.
5. **Примеры и доки.** Обновить `examples/tabular_hidden_states/td_data_collection.ipynb`
   и `examples/basics/tabular_preprocessing.ipynb` — показать local-путь без
   SparkSession. Раздел в `docs/data/`.
6. (Опц.) polars bump + путь B, если профиль покажет необходимость.

Фазы 1–2 самодостаточны и ничего не ломают в Spark-пути.
