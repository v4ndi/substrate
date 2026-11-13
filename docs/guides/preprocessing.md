# Предобработка

Модели не читают сырые таблицы. Препроцессор превращает произвольный parquet в
две упакованные колонки, которые понимают датасеты:

* `cat_features` — `list<int64>`, идентификаторы категорий со **сквозной**
  нумерацией, чтобы все категориальные колонки делили один `nn.Embedding`;
* `num_features` — `list<float32>`, стандартизованные числовые признаки.

Для событийных последовательностей выход другой: одна строка на клиента, каждый
атрибут — список, упорядоченный по времени.

## Два бэкенда, один API

| | `avatar.preprocessing.spark` | `avatar.preprocessing.local` |
|---|---|---|
| движок | PySpark на кластере | pyarrow + numpy, одна машина |
| требует | Spark / YARN / JVM (JDK 8/11/17) | ничего сверх базовых зависимостей |
| память | распределённо | потоково, ограниченно — переваривает данные больше RAM |
| масштаб | любой | ~8 CPU / 128 ГБ, до ~100 млн строк × ~500 признаков |
| порядок id категорий | `collect_set`, недетерминированный | детерминированный |

Классы называются одинаково: `TabularPreprocessor` и
`EventSequencePreprocessor` есть в обоих пакетах, с одинаковыми аргументами.

**Артефакты переносятся между бэкендами в обе стороны.** Обучили на Spark —
загрузили в local и наоборот; результат `transform` совпадает. Это проверяется
запускаемыми примерами:

```bash
python examples/tabular_preprocessing/generate_data.py
python examples/tabular_preprocessing/run_both_backends.py
```

## Что делает препроцессор

**Категориальные колонки** — `LabelEncoder`: `значение → id` по каждой колонке,
неизвестное значение и null → `unk` (0). Затем строится накопительный
`offset_map`, сдвигающий id каждой колонки так, чтобы диапазоны не пересекались.
Именно поэтому в конфиге модели `vocab_size` — одно число на все колонки.

**Числовые колонки** — `StandardScaler`: опциональный `signed_log1p` для колонок
из `to_log_columns`, затем `(x - mean) / (std + 1e-8)`, затем null → `0.0`.

`identity_cols` переносятся в выход как есть, рядом с упакованными массивами —
так в датасет попадают идентификатор клиента, таргет, дата отчёта.

## Типовое использование

```python
import yaml
from avatar.preprocessing.local import TabularPreprocessor

pp = TabularPreprocessor(
    categorical_columns=cat_cols,
    numeric_columns=num_cols,
    spec_tokens={"pad": 0},
    label_encoder_kwargs={"frequency_encoder": True},
    standard_scaler_kwargs={"to_log_columns": log_cols},
)
pp.fit("/data/train")                                    # один потоковый проход
pp.transform("/data/train", "/data/train_processed", identity_cols=["epk_id", "target"])
pp.transform("/data/valid", "/data/valid_processed", identity_cols=["epk_id", "target"])
pp.transform("/data/test",  "/data/test_processed",  identity_cols=["epk_id", "target"])

yaml.safe_dump(pp.dump(), open("artifacts/preprocessor.yaml", "w"))
```

**Обучать препроцессор нужно только на train.** Для valid и test вызывается
`transform` тем же объектом — иначе статистики протекут между выборками и метрики
станут оптимистичными.

`fit` принимает путь, glob, каталог, список любого из этого или
`pyarrow.dataset.Dataset`. `transform` пишет parquet, если задан `output_path`,
иначе возвращает `pyarrow.Table`.

## Артефакт и инференс

```python
pp = TabularPreprocessor.load(yaml.safe_load(open("artifacts/preprocessor.yaml")))
pp.transform("/data/new", "/data/new_processed")
```

Артефакт нужно хранить рядом с весами модели: без **того же самого** отображения
`значение → id` веса эмбеддингов бессмысленны. Это самая частая причина, по
которой модель, отлично работавшая на валидации, выдаёт мусор на новых данных.

## Три числа для конфига модели

После обучения препроцессора:

| параметр конфига | откуда |
|---|---|
| `vocab_size` | `pp.vocab_size` |
| `num_numerical_features` | длина `num_features`, то есть `len(num_cols)` |
| `aggregation_config.num_features` | `len(cat_cols) + len(num_cols)`, плюс 1 за каждый внешний эмбеддинг, подмешанный как признак |

Их же печатает `examples/tabular_preprocessing/run_both_backends.py`.

## Событийные последовательности

Вход — одна строка на событие. `fit` учит те же статистики (проход общий с
табличным пайплайном). `transform`:

1. кодирует категориальные колонки и стандартизует числовые — **без** сквозного
   `offset_map`: последовательностная модель использует отдельные эмбеддинги на
   колонку, поэтому id остаются локальными для колонки;
2. сортирует события по `(id, время события)`;
3. масштабирует время в `unix_seconds / {days|weeks|months}` (float32);
4. группирует по `id` и собирает каждый атрибут в список.

Выход — одна строка на клиента, каждый признак — колонка-список.

### Порядок событий различается между бэкендами

`sort().groupBy().collect_list()` в Spark **не гарантирует** порядок: shuffle при
group-by теряет сортировку. На малых данных обычно получается отсортированно, но
не всегда. Локальный бэкенд сортирует по исходной временной метке полной
точности и всегда монотонен.

Если порядок событий важен для вашей модели — а для next-k предсказания он
критичен, — используйте локальный бэкенд либо проверяйте монотонность на
выходе.

## Настройка под большие данные

* `batch_rows` (по умолчанию 250 000) — строк в потоковом чанке; пик памяти
  примерно `batch_rows × n_columns × 8 Б`.
* `max_cardinality` и `on_overflow="topk"` у label-энкодера — защита от
  идентификаторо-подобных колонок, которым нужен не словарь, а
  `HashEmbedding`.
* `frequency_encoder=True` — присваивать id по убыванию частоты. Делает порядок
  детерминированным и ставит частые значения в начало таблицы.

## Куда дальше

* [datasets.md](datasets.md) — как обработанные данные читаются в обучении.
* [../getting_started.md](../getting_started.md) — весь путь целиком.
* `examples/tabular_preprocessing/` и `examples/eventsequence_preprocessing/` —
  запускаемые примеры на синтетике.
