# TODO

## Переход табличного препроцессинга со Spark на уровень датасета (torch/GPU)

### Контекст

Пример `examples/tabular_hidden_states` собирает данные и обучает табличный
трансформер (`avatar.nn.tabular.STEv2`) с external embedding, используя
`avatar.preprocessing.spark.TabularPreprocessor` для препроцессинга.

Цель — отказаться от Spark-`transform` и делать препроцессинг на torch, перекладывая
тензоры на GPU. Дополнительно — не собирать parquet с упакованными колонками
`cat_features` / `num_features`, а подавать на вход широкую табличку (одна колонка на
признак).

Текущий препроцессинг полностью статический и описывается маленьким словарём
статистик (`TabularPreprocessor.dump()`): `values_to_id`, `mean_std`, `offset_map`,
`vocab_size`, `to_log_columns`, `spec_tokens`.

- `LabelEncoder`: per-column `value -> id`, неизвестное/NULL -> 0 (`unk`).
- `StandardScaler`: опц. `signed_log1p` -> `(x - mean) / (std + 1e-8)` -> `fillna(0)`.
- `TabularPreprocessor`: кумулятивный `offset_map`, чтобы все категориальные колонки
  жили в одной `nn.Embedding`; на выходе `cat_features: array<long>` (со смещениями),
  `num_features: array<float>`.

Пример данных: 52 категориальных колонки (мелкие целые коды, значения `-1..6`),
`vocab_size=172`; 189 числовых; `to_log_columns` пуст; `fillna=True`.

### Оценка: утилизация GPU

Сохранится. Арифметика `transform` на GPU ничтожна (gather/add по `cat` + аффинное
преобразование `num`, ~1M элементов на батч 2048xN) против forward/backward
трансформера. Перенос «тяжёлого» из collate на GPU скорее разгружает CPU-воркеры.

Утилизацию определяет не арифметика, а:
1. пропускная способность чтения parquet + декомпрессия (не меняется, `num_workers`
   так же нужны);
2. host->device transfer (payload ~тот же; можно уменьшить: `cat` в int16/int32,
   `num` в fp16);
3. CUDA в DataLoader-воркерах невозможна -> GPU-препроцессинг живёт в main-процессе
   после `move_to_device` (`avatar/train/loop.py`, `Trainer._run_epoch`), т.е. в
   начале `forward` пайплайна;
4. строковые категории: маппинг строка->id по батчу на CPU убьёт утилизацию ->
   категории должны приходить integer-кодами (факторизация в fit-шаге).

### Главный риск широкой таблички

`avatar/data/parquet.py:read_parquet_file` отдаёт Python-dict на строку, по ключу на
колонку (`for row in zip(*col_arrays)`). Сейчас это 2 колонки-массива, с широкой
табличкой — 241 колонка => в ~100x больше словарной возни на строку.

Нужна векторная сборка: прочитать Arrow-таблицу файла/row-group и один раз сделать
`np.column_stack` по num- и cat-колонкам в два 2D-массива, без прохода по строкам с
241 полем.

### План: гибрид (fit оффлайн, transform на GPU)

1. **Лёгкий fit-шаг** (Spark или polars/duckdb по выборке) считает те же статистики и
   переиспользует формат `TabularPreprocessor.dump()` — inference-артефакты и
   совместимость с РХ не ломаются, нет train/serving skew (ср. `[[fmlib-mtl-pr19]]`
   про bit-identical inference).

2. **`WideTabularDataset(IterDataset)`** — новый датасет:
   - принимает списки `categorical_columns` / `numeric_columns` (порядок жёстко из
     артефакта) + `hidden_state_columns`;
   - переопределяет чтение файла: `table = ds.dataset(file).to_table(columns=...)`,
     затем `num_block = np.column_stack([...]).astype("float32")` (nullable -> NaN),
     `cat_block = np.column_stack([...]).astype("int64")` (NULL -> сентинел, напр.
     `-999`);
   - шифл строк — перестановкой индексов по блокам (как `shuffle_pq`) или шифл-буфером;
   - hidden states, target, id — как сейчас;
   - `process` / collate дают тот же `TabularBatch`, но **сырой**: без offset, без
     скейлинга.

3. **`TorchTabularTransform(nn.Module)`** с буферами из артефакта (`register_buffer`:
   `offset` `[N_cat]`, `mean` `[N_num]`, `std` `[N_num]`, `log_mask` `[N_num]`, LUT /
   отсортированные значения для cat), вызывается в начале `forward` пайплайна после
   `move_to_device`:
   - cat: `id = lut[cat_raw]` (для этих данных LUT маленькая) либо `searchsorted` по
     отсортированным значениям (общий случай); неизвестное/сентинел -> `unk`;
     `+ offset_map`;
   - num: `where(log_mask, signed_log1p(x), x)` -> `(x - mean) / (std + 1e-8)` ->
     `nan_to_num`;
   - буферы едут на device через `accelerator.prepare`.

4. **parquet** пишет сырые широкие колонки, без упаковки. Категории обязательно
   integer codes.

5. **Побочно**: переписать `TabularDataset.process_tabular` на поблочную конвертацию
   Arrow row-group -> тензор вместо `np.array` на строку; добавить `non_blocking=True`
   в `move_to_device` при `pin_memory=True`.

6. **Бенчмарк**: `ShardTabularDataset.configure_epoch_metrics` / `get_epoch_metrics`
   (`parquet_bytes_opened`, `filter_sec`) — сравнить до/после.

### Подводные камни

- бит-точность математики (`log1p`, `+1e-8`, обработка `unk`/NaN) — тест на совпадение
  со Spark-выходом;
- `nn.Embedding` ждёт индексы в `[0, vocab_size)` — клип неизвестных категорий
  обязателен на этапе transform;
- per-column приведение типов: nullable int в Arrow приходит маской -> явный каст
  cat-блока, NULL -> unk;
- дрейф схемы / отсутствующие колонки — датасет должен падать явно или подставлять
  дефолт;
- порядок колонок жёстко из артефакта — layout тензора стабилен train/inference;
- row-group не должны быть мелкими (иначе `column_stack` по многим крошечным чанкам);
- фичи, требующие групповых статистик на инференсе (target/frequency encoding по
  свежим данным) — всё равно оффлайн;
- multi-gpu: буферы-статистики одинаковы на всех рангах (грузятся из артефакта).

### Опция B (выше throughput, больше вмешательства)

Датасет отдаёт сразу батчи (`DataLoader(batch_size=None)`), collate — passthrough.
Убирает per-row overhead полностью, но меняет семантику шифла (нужен шифл-буфер) и
трогает train-loop / accelerate. Начинать с основного варианта, к B идти по профилю.

---

## Категориальный эмбеддинг-слой (`avatar.nn.embedding.TabularEmbedding`)

Категориальная часть — одна общая `nn.Embedding(vocab_size, hidden_size)` на
глобально смещённых id (`offset_map` даёт каждой колонке непересекающийся диапазон).
`STEv2Block` — чистый self-attention + FFN, **без позиционного кодирования**.

### Оставить как есть (для текущих данных — 52 низкокардинальных колонки, vocab 172)

Общая таблица + смещения — самый дешёвый путь (один gather-кернел, без Python-цикла
по колонкам). Под миграцию на GPU удобно: cat-кодирование = `searchsorted`/LUT ->
локальный id -> `+ offset` (буфер `[N_cat]`), фьюзнутый оп, выход прямо в
`nn.Embedding`.

### Улучшения (сделать заодно с рефакторингом transform)

1. **`padding_idx=0`** в `nn.Embedding` (сейчас `nn_embedding_config={}`; id 0 = `pad`
   зарезервирован, но мёртв — в него ничего не маппится, unk идёт со смещением).
2. **Защитный `clamp(0, vocab_size - 1)`** перед gather (сейчас сырой gather; с трюком
   смещений легко получить id вне диапазона -> device-side assert). `TorchTabularTransform`
   всё равно обязан гарантировать `0 <= id < vocab_size`.
3. **Обучаемый per-feature эмбеддинг** `nn.Parameter(N_cat + N_num, hidden_size)`,
   добавляемый к `combined` — компенсирует отсутствие позиционного сигнала в
   `STEv2Block` (особенно важно: 189/241 фичи числовые, у них только per-feature веса
   в `LinearEmbeddings`; низкокардинальные cat-колонки дают почти константу).
   **Прогнать абляцию** — вероятно, даст больше, чем возня с самим кодированием.
4. **`torch.randn_like(combined)`** вместо `torch.randn(combined.size()).to(device)` в
   ветке `std_noise` (сейчас создаётся на CPU и копируется). Также пересмотреть,
   нужен ли шум на категориальных эмбеддингах вообще.

### Известные ограничения дизайна (не блокеры, задокументировать)

- эмбеддинг бессмыслен без точного `offset_map`; добавление одной категории в одну
  колонку сдвигает id всех последующих -> полный retrain;
- общая таблица: одна шкала init / один weight decay на бинарную и на потенциально
  высококардинальную колонку;
- per-column `nn.Embedding` (`ModuleDict`) имеет смысл только при высокой кардинальности
  или частом дрейфе словаря — тогда лучше прямоугольная таблица
  `(N_cat, max_card, hidden_size)` с одним gather, а не Python-цикл.
