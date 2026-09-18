# `EventSequenceBatch`  

Класс для работы с батчами временных последовательностей событий.  

**Модуль**: `fmlib.data.EventSequenceBatch`  

## Описание  

Класс представляет батч временных последовательностей событий (например, клики пользователей, транзакции) с метаданными и методами для анализа. Поддерживает:  
- Различные типы событий в одной последовательности  
- Временные метки событий  
- Маски валидности (для обработки padded-последовательностей)  
- Конвертацию в `PaddedBatch`  

---

## Инициализация  

```python
def __init__(
    events: dict[str, torch.Tensor],
    timestamps: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    event_ids: Optional[torch.LongTensor] = None
)
```

**Параметры**:  
  
| Параметр         | Тип                          | Описание                                                                        |  
|------------------|------------------------------|---------------------------------------------------------------------------------|  
| `events`         | `dict[str, torch.Tensor]`    | Словарь с тензорами событий (формат: `(batch_size, seq_len, )`)                 |  
| `timestamps`     | `Optional[torch.FloatTensor]`| Тензор временных меток `(batch_size, seq_len)`.                                 |  
| `attention_mask` | `Optional[torch.LongTensor]` | Маска валидности `(batch_size, seq_len)`. `1` – валидное событие, `0` – паддинг |  
| `event_ids`      | `Optional[torch.LongTensor]` | Идентификаторы типов событий `(batch_size, seq_len)`                            |  

**Исключения**:  
- `ValueError` – **если размеры тензоров не согласованы**  

---

## Атрибуты  

### Основные  

| Атрибут           | Тип                           | Описание                     |
|-------------------|-------------------------------|------------------------------|  
| `_events`         | `dict[str, torch.Tensor]`     | Словарь тензоров событий     |  
| `_timestamps`     | `Optional[torch.FloatTensor]` | Тензор временных меток       |  
| `_attention_mask` | `Optional[torch.LongTensor]`  | Маска валидности             |  
| `_event_ids`      | `Optional[torch.LongTensor]`  | Идентификаторы типов событий |  

### Вычисляемые свойства  
```python
@property
def seq_len() -> torch.LongTensor
```  
**Возвращает**:  
- Тензор длин последовательностей `(batch_size,)` (сумма по `attention_mask`)  

```python
@property
def sequence_columns() -> list[str]
```  
**Возвращает**:  
- Список названий типов событий (ключи `_events`)  

```python
@property
def device() -> torch.device
```  
**Возвращает**:  
- Устройство (CPU/GPU), на котором расположены тензоры  

---

## Основные методы  

### Доступ к данным  
```python
def __getitem__(self, key: str) -> torch.Tensor
```  
**Пример**:  
```python
click_events = batch["click"]  # shape=(batch_size, seq_len,)
```

### Временные характеристики  
```python
def get_timedeltas(self, step: int = 1) -> torch.FloatTensor
```  
**Параметры**:  
- `step` – шаг для вычисления дельты (по умолчанию 1)  

**Возвращает**:  
- Тензор разниц во времени `(batch_size, seq_len)` с паддингом нулями  

**Пример**:  
```python
deltas = batch.get_timedeltas(step=2)  # интервалы между событиями через 1
```

### Фильтрация событий  
```python
def event_attn_mask(self, event_id: Union[int, list[int]] = None) -> torch.LongTensor
```  
**Параметры**:  
- `event_id` – ID или список ID событий для фильтрации  

**Возвращает**:  
- Маску, где `1` только для указанных типов событий  

**Пример**:  
```python
mask = batch.event_attn_mask(event_id=[1, 3])  # маска по событиям типов 1 и 3
```

### Агрегация  
```python
def num_items(self, event_id: int = None) -> int
```  
**Возвращает**:  
- Общее количество событий указанного типа (или всех, если `event_id=None`) без учета `pad` токенов  

---

## Утилиты  

### Перемещение на устройство  
```python
def to(self, device: torch.device) -> EventSequenceBatch
```  
**Пример**:  
```python
batch = EventSequenceBatch(...)
batch_gpu = batch.to("cuda:0")
```

---

## Примеры использования  

### Создание батча
### Последовательность транзакций  
```python
import torch
from fmlib.data import EventSequenceBatch


events = {
    "mcc": torch.LongTensor(
        [
            [1, 5, 7, 2, 0, 0],  # 0 - pad token
            [2, 3, 4, 1, 5, 7],  # 0 - pad token
        ]),
    "price": torch.FloatTensor(
        [
            [123.7, 1000.24, 10.5, 20., 0., 0.],  # 0. - pad value
            [70.13, 30.85, 40., 110., 50., 77.],
        ]),
}
timestamps = torch.FloatTensor(  # UnixTimeStamp в днях
    [
        [123., 123.5, 124.1, 124.2, 0., 0.],  # 0. - pad value
        [70.13, 70.85, 71.2, 73., 73.1, 74.12],
    ]
)
attention_mask = torch.LongTensor([
    [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1]  # 0 если событие нет в последовательности на этом месте
    ]
])

batch = EventSequenceBatch(events=events, timestamps=timestamps, attention_mask=attention_mask)
```

### Последовательность транзакций и кликстрим
```python
import torch
from fmlib.data import EventSequenceBatch


events = {
    "txn_mcc": torch.LongTensor( # Категориальный признак
        [
            [1, 0, 7, 0, 0, 0], # 0 - pad token
            [2, 3, 4, 0, 5, 7], # 0 - pad token
        ]),
    "txn_price": torch.FloatTensor(  # Непрерывный признак
        [
            [123.7, 0., 24., 10.5, 0., 0., 0.], # 0. - pad value
            [70.13, 30.85, 0., 110., 50., 77., 35.2], # 0. - pad value
        ]),
    "clickstream_category": torch.LongTensor([
        [0, 3, 0, 7, 0, 0],
        [0, 0, 0, 2, 0, 0]
    ])
}

event_ids = torch.LongTensor([  # 0 - txn, 1 - clickstream, -1 - события не существует
    [0, 1, 0, 1, -1, -1],
    [0, 0, 0, 1, 0, 0],
])

timestamps = torch.FloatTensor(  # UnixTimeStamp в днях
    [
        [123., 123.5, 124.1, 124.2, 0., 0.], # 0. - pad value
        [70.13, 70.85, 71.2, 73., 73.1, 73.3], # 0. - pad value
    ]
)

attention_mask = torch.LongTensor([
    [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1]  # 0 если атрибута нет в последовательности на этом месте
    ]
])
```

#### **event_attn_mask**
По типу или типам событий возвращает бинарную маску наличия события в последовательности.
  
**Пример:**
Код ниже применяется к последовательности транзакций и кликстрима, описанных выше 
```python
batch = EventSequenceBatch(
    events=events,
    timestamps=timestamps,
    attention_mask=attention_mask,
    event_ids=event_ids
)

batch.event_attn_mask(event_id=0)  # Маска по событиям транзакций
# tensor([[1, 0, 1, 0, 0, 0],
#         [1, 1, 1, 0, 1, 1]])

batch.event_attn_mask(event_id=1) # Маска по событиям из кликстрима
# tensor([[0, 1, 0, 1, 0, 0],
#         [0, 0, 0, 1, 0, 0]])

batch.event_attn_mask(event_id=[0, 1]) # Маска по событиям из кликстрима и транзакций, эквивалентна attention_mask
# tensor([[1, 1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 1, 1]])
```


```python
>>> batch = EventSequenceBatch(events=events, timestamps=timestamps, attention_mask=attention_mask)
>>> batch.event_attn_mask(event_id=1)
>>> torch.LongTensor
```

#### num_items  
```python
batch.num_items(1) # return 3 - всего три события кликстрима в батче
```