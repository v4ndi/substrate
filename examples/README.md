# Examples

## Навигация

* `basics` — предобработка табличных данных, подгрузка их в датасет, создание собственной метрики
* `tabular_preprocessing` — табличный препроцессинг на обоих бэкендах (Spark и local),
  кросс-загрузка артефакта `dump()`/`load()` из одного в другой
* `eventsequence_preprocessing` — то же для препроцессинга событийных последовательностей
* `tabular_hidden_states` — обучение табличного трансформера на классификацию,
  late/early fusion при работе с эмбеддингами других нейросетей
* `next_event_prediction` — self-supervised обучение на событийных последовательностях (next-token / next-k)
* `uplift_modeling/s_learner` — обучение модели в uplift-постановке (S-Learner) с использованием hidden_states

## TODO

* EventSequence SSL (NextToken, NextK, MLM) — расширить пример
* Запуск на batch_datalab / supercomp
