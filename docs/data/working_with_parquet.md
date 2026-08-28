# Работа с .parquet файлами

Модуль `avatar.data.parquet` предоставляет инструменты для эффективной работы с файлами формата Parquet.

## Основные возможности
- Получение количества строк без чтения всего файла
- Чтение данных с выбором конкретных колонок
- Поддержка случайного перемешивания данных
- Итеративная обработка для работы с большими файлами

## Функции модуля

### `parquet_num_rows(path: str) -> int`
```python
def parquet_num_rows(path: str) -> int:
    """Returns the number of rows in a parquet file or dataset.

    Args:
        path: Path to the parquet file or directory containing parquet files.
              Can be either a string or Path object.

    Returns:
        Number of rows in the parquet file(s). For directories, returns total
        row count across all files in the dataset.
    """
    with pq.ParquetFile(path) as f:
        return f.metadata.num_rows
```
  
Возвращает количество строк в Parquet-файле или наборе файлов, используя только метаинформацию (без чтения всего файла).

**Параметры:**
- `path` (str или Path) - Путь к Parquet-файлу или директории с набором файлов

**Возвращает:**
- `int` - Общее количество строк (для директорий - сумма по всем файлам)

**Пример использования:**
```python
>>> from avatar.data.parquet import parquet_num_rows
>>> parquet_num_rows("data.parquet")
>>> 1011
```

**Примечания:**
- Для пустых файлов возвращает 0
- Поддерживает как отдельные файлы, так и партиционированные наборы
- Работает очень быстро, так как не загружает данные

---

### `read_parquet_file(file: str, columns: Optional[List[str]] = None, shuffle: bool = True) -> Iterator[Dict[str, Any]]`
```python
def read_parquet_file(
    file: str, columns: Optional[List[str]] = None, shuffle: bool = True
) -> Iterator[Dict[str, Any]]:
    """Reads a parquet file and yields records as dictionaries with optional shuffling.

    Args:
        file: Path to the parquet file to read
        columns: Optional list of column names to read. If None, reads all columns.
        shuffle: Whether to randomly shuffle the rows before yielding. Defaults to True.

    Yields:
        Dictionary for each row, where keys are column names and values are numpy arrays
        containing the row values. Note that scalar values will still be returned as
        numpy arrays (use .item() to convert to Python scalar if needed).

    Note:
        - When shuffle=True, the entire file is loaded into memory temporarily
        - For very large files, consider using shuffle=False or specifying columns
        - Numpy arrays are returned even for scalar values for consistency

    Example:
        >>> # Read specific columns with shuffling
        >>> for record in read_parquet_file("data.parquet", columns=["id", "value"]):
        ...     print(record["id"].item(), record["value"].item())
        >>>
        >>> # Read all columns without shuffling
        >>> for record in read_parquet_file("data.parquet", shuffle=False):
        ...     print(record)
    """
    table = pq.read_table(file, use_threads=False, columns=columns)

    if columns is None:
        columns = [col._name for col in table.columns]

    if shuffle and table.num_rows != 0:
        row_indexes = [i for i in range(table.num_rows)]
        np.random.shuffle(row_indexes)
        table = table.take(row_indexes)

    for rb in table.to_batches():
        col_arrays = [rb.column(x) for x in columns]
        col_arrays = [x.to_numpy(zero_copy_only=False) for x in col_arrays]
        for row in zip(*col_arrays):
            record = {}
            for col, arr in zip(columns, row):
                record[col] = arr
            yield record
```
  
Читает Parquet-файл и возвращает записи в виде словарей с возможностью перемешивания.

**Параметры:**
- `file` (str) - Путь к Parquet-файлу
- `columns` (List[str], optional) - Список колонок для чтения (по умолчанию все)
- `shuffle` (bool) - Флаг случайного перемешивания строк (по умолчанию True)

**Возвращает:**
- `Iterator[Dict[str, Any]]` - Итератор по строкам файла

**Примеры использования:**
```python
# Чтение всех колонок с перемешиванием

for record in read_parquet_file("data.parquet"):
    print(record)

# Чтение конкретных колонок без перемешивания
for record in read_parquet_file("data.parquet", 
                              columns=["id", "timestamp"],
                              shuffle=False):
    print(record["id"].item(), record["timestamp"].item())
```

**Особенности работы:**
- Все значения возвращаются как numpy-массивы для единообразия
- Для получения скалярных значений используйте `.item()`

**Производительность:**
- Для ускорения работы указывайте только нужные колонки
  