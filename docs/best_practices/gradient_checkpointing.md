## Gradient Checkpointing

Тут указано как включить gradient checkpointing в core_fm

### Описание

gradient_checkpointing помогает сэкономить память, путем хранения меньшего количества активаций и их пересчета (вместо хранения) на этапе backward'а

### Как добавить в конфиг
1. Добавить в блок `ddp:` данные 3 строки (некоторые мб лишние, но рекомендуются все, static_graph: True обязателен)
```
ddp:
  find_unused_parameters: False
  broadcast_buffers: False
  static_graph: True
```

2. Добавить в GPT2Config строки (обе строки обязательны):
```
_target_: transformers.GPT2Model
config:
  _target_: transformers.GPT2Config
  ...
  ...
  gradient_checkpointing: True
  use_cache: False
  ...
  ...
```


### Где это стояло

Обе правки — из конфига обучения на событийных последовательностях: HF-бэкенд
(`transformers.GPT2Model`) под `fmlib.nn.sequential.TransformersWrapper`,
батчи по 384–440 последовательностей длиной до 512 событий. Полный конфиг того
запуска не приводится: пайплайн, который в нём стоял, из репозитория удалён
(см. `docs/decisions/pipeline_boundaries.md`), а сами три настройки от
пайплайна не зависят — они относятся к DDP и к HF-конфигу.

`static_graph: True` обязателен: без него DDP на каждом шаге заново ищет
неиспользованные параметры, а пересчёт активаций как раз меняет набор
использованных.


### Результаты:
Значения batch_size подобраны так, чтобы финальная утилизация составила 38.5 GB на описанной выше модели:


| t                           | baseline                       | grad checkpoint wit big batch  |
|-----------------------------|--------------------------------|--------------------------------|
| `batch_size`                | 384                            | 440 (+15%)                     |
| `gradient_checkpointing`    | False                          | True                           |
| `время на эпоху`            | 250 min                        | 270 min (+8%)                  |

