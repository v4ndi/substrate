# Next Event Prediction

![](artifacts/avatar_fm_next_token.jpg)

Self-supervised предобучение на событийных последовательностях. Модель учится
предсказывать следующие события, а полученный backbone потом используется для
выгрузки эмбеддингов клиента (`seq_hidden_state`), которые читают табличные
модели.

---

## Навигация

* `next_event_prediction.ipynb` — сбор и подготовка данных для обучения
* `configs/ssl.yaml` — next-token: предсказание одного следующего события
* `configs/ssl_next_k.yaml` — next-K: предсказание трёх следующих событий
* `configs/inference.yaml` — выгрузка эмбеддингов клиента
* `download_data.sh` — скрипт подгрузки данных из РХ

---

## Запуск обучения

```bash
chmod +x download_data.sh
./download_data.sh
```

```bash
torchrun --standalone --nproc_per_node=1 -m avatar.train --config-dir=configs/ --config-name=ssl
```

## Next-token против next-K

Обе постановки — это один и тот же пайплайн
`avatar.pipeline.sequence.NextKTokensPrediction`; разница только в конфиге.
`configs/ssl_next_k.yaml` наследует `configs/ssl.yaml` через `defaults:` и
меняет несколько ключей:

```bash
torchrun --standalone --nproc_per_node=1 -m avatar.train \
    --config-dir=configs/ --config-name=ssl_next_k
```

| ключ | что делает |
|---|---|
| `horizon` | на сколько событий вперёд предсказываем; `1` — это и есть next-token |
| `horizion_loss_weight` | дисконт дальних горизонтов: `loss_k = loss_k / k ** weight`; `0` — все горизонты равны |
| `feature_loss_weights` | вес на признак, применяется **только при обучении** |
| `enable_event_id_prediction` | предсказывать ещё и тип события и подавать его эмбеддинг обратно в следующий горизонт |

Больший горизонт — более сильная задача: модель вынуждена держать в
представлении не только ближайшее событие. Платить за это приходится тем, что
голов становится `horizon × число признаков`, и шаг обучения дорожает.

`horizion_loss_weight` стоит трогать, если дальние горизонты доминируют в
потерях: они предсказываются хуже, поэтому их вклад больше, и без дисконта
модель может оптимизировать в основном их.

## Что смотреть в метриках

`avatar.metrics.MultiLossMetric` разбивает суммарные потери на компоненты — по
одной на пару «признак × горизонт», с именами вида `evt_attr_9_head_0`.
Полезно именно это, а не общая сумма: по компонентам видно, какая голова
перестала учиться.

Компоненты взвешиваются по числу валидных элементов каждой головы, поэтому
головы с разным количеством непаддинговых позиций остаются сравнимыми.

## MLM пока не реализован

Постановка masked language modelling в репозитории отсутствует: подходящего
пайплайна нет, и одним конфигом её не выразить. Это задача на разработку
модели, а не на настройку.

## Запуск инференса

```bash
python -m avatar.infer --config-dir=configs --config-name=inference
```

По итогу будут сохранены `.parquet` файлы со следующими полями:

* `epk_id` — идентификатор клиента;
* `seq_hidden_state` — эмбеддинг клиента.

Если нужно сохранить дополнительные поля, укажите список колонок в
`metrics.test_metrics.additional_columns`.

Инференс использует `avatar.pipeline.sequence.SequenceModelWithAggregation`:
тот же backbone, но без голов предсказания, с агрегацией скрытых состояний в
один вектор. Веса подгружаются через `model_weights`.

## Куда дальше

* [../../docs/guides/models.md](../../docs/guides/models.md) — как собирается
  последовательностный стек.
* [../tabular_hidden_states/](../tabular_hidden_states/) — как использовать
  полученные эмбеддинги в табличной модели.
