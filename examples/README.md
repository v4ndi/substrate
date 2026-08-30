# Примеры

Сквозные примеры использования. Первое, что стоит проверить перед запуском, —
колонку «данные»: половина примеров работает на синтетике и запускается сразу,
остальным нужна выгрузка из внутреннего хранилища.

| пример | чему учит | формат | данные |
|---|---|---|---|
| [`tabular_preprocessing`](tabular_preprocessing/) | табличный препроцессинг на обоих бэкендах, перенос артефакта между ними | скрипт | синтетика, генерируется скриптом |
| [`eventsequence_preprocessing`](eventsequence_preprocessing/) | то же для событийных последовательностей | скрипт | синтетика, генерируется скриптом |
| [`distributed_training`](distributed_training/) | multi-GPU и multi-node обучение, шардирование, AMP, DDP | скрипт + конфиг | синтетика, генерируется скриптом |
| [`custom_callback`](custom_callback/) | своя точка расширения в цикле обучения | модуль + тесты | не нужны |
| [`custom_loss`](custom_loss/) | своя функция потерь, подключаемая из конфига | модуль + тесты | не нужны |
| [`basics`](basics/) | загрузка данных в датасет, своя метрика | ноутбуки | внутренние (HDFS) |
| [`tabular_hidden_states`](tabular_hidden_states/) | обучение табличного трансформера, early/late fusion внешних эмбеддингов | конфиги + ноутбуки | скрипт выгрузки |
| [`next_event_prediction`](next_event_prediction/) | self-supervised обучение на событиях, выгрузка эмбеддингов клиента | конфиги + ноутбук | скрипт выгрузки |
| [`uplift_modeling/s_learner`](uplift_modeling/s_learner/) | uplift-постановка (S-Learner) поверх скрытых состояний | конфиги + ноутбук | скрипт выгрузки |

## Запускается прямо сейчас

```bash
# препроцессинг на обоих бэкендах
python examples/tabular_preprocessing/generate_data.py
python examples/tabular_preprocessing/run_both_backends.py

# обучение на нескольких процессах
python examples/distributed_training/generate_data.py
# скрипт напечатает готовую команду torchrun

# точки расширения
python -m pytest tests/examples
```

Примеры препроцессинга и распределённого обучения генерируют данные сами и не
требуют ни Spark, ни доступа к хранилищу: Spark-часть пропускается, если
подходящей JVM нет, а распределённое обучение работает и на CPU через gloo.

## Требует выгрузки данных

Три примера читают подготовленные данные из внутреннего хранилища. В каталоге
каждого из них лежит свой скрипт выгрузки, запускать его нужно оттуда же:

```bash
kinit
cd examples/tabular_hidden_states
chmod +x download_data.sh
./download_data.sh
```

После выгрузки в конфигах нужно прописать абсолютные пути до данных.

## Формат примеров

Всё, что должно продолжать работать, оформляется скриптом `.py` плюс README.
Ноутбуки остаются для повествования — они исключены из ruff и pre-commit, а
значит, не могут быть проверены автоматически и не должны нести на себе
критичное использование API.

## Куда смотреть дальше

* [../docs/getting_started.md](../docs/getting_started.md) — путь от установки
  до инференса.
* [../docs/configuration/schema.md](../docs/configuration/schema.md) — все ключи
  конфига.
* [../docs/README.md](../docs/README.md) — навигация по документации.

## TODO

* EventSequence SSL (NextToken, NextK, MLM) — расширить пример
* Запуск на batch_datalab / supercomp
