# Проведение пром/пилот сценария
Для запуска обучения/инференса необходимо установить все дополнительные зависимости, подробная инстуркция есть в корневой директории репозитория.

> **Чтение с HDFS.** Датасеты умеют читать parquet прямо из РХ — в `path` можно
> указать `hdfs://...` вместо локальной директории, и шаг `hdfs dfs -get` тогда не
> нужен. Требуется libhdfs в окружении (`HADOOP_HOME`/`ARROW_LIBHDFS_DIR` +
> `CLASSPATH`); каждый DataLoader-воркер поднимает свою JVM, поэтому при работе с
> РХ следите за `num_workers`. Ниже оставлен вариант с предварительным копированием
> на локальный диск.

---
  
## Инференс/Обучение пром сценария
### Инференс sequence данных
1. Подгрузка данных из РХ
```bash
hdfs dfs -get <PASS YOUR SEQUENCE_DATA_PATH>
```
2. Подгрузка модели из РХ
```bash
hdfs dfs -get /user/team/team_ai_avatar/rusakov/september_pilot/prom/model_weights/pilot_seq_model.bin
```
3. Файл конфигурации `configs/prom/inference_sequence.yaml`
* `test_dataloader.datset.path` - указать путь до директории из пункта 1
* `model.model_weights` - указать путь до весов модели из 2 пункта
* `metrics.test_metrics.path_to_save` - указать свою директорию, в эту директорию будут сохраняться `parquet` файлы с результатами.
4. Запуск инференса
```bash
python -m avatar.inference --config-dir=configs/prom/sequence --config-name=inference_sequence
```
5. Выгрузка эмбеддингов в РХ
```bash
hdfs dfs -put path_to_dir path_to_hdfs
```
  
**При инференсе всей базы клиентов, время инференса займет 24 часа на одной карте. Для ускорения можно разделить выборку на две части и запустить инференс паралелльно на двух картах**.

## Обучение модели по каждому продукту
Файлы конфигураций по каждому продукту лежат в директории - `config/prom/tabular/train/`. Перед запуском обучения необходимо заполнить следующие поля:
* `train_dataloader.dataset.path` - путь до директории с данными для обучения
* `valid_dataloader.dataset.path` - путь до директории с данными для валидации
  
**Запуск обучния:**
```bash
accelerate launch -m avatar.train --config-dir=configs/prom/tabular/train/ --config-name=PASS_YOUR_PRODUCT_NAME_CONFIG
```

## Инференс продуктов общий случай
Инференс всех продуктов происходит на одном и том же наборе данных, но разными моделями. Из за того, что кол-во каналов коммуникаций в разных продуктах разное, время инференса отличается.   
**Файлы конфигураций** - `configs/prom/tabular.inference/*.yaml`. Для запуска необходимо указать:
* `test_dataloader.dataset.path` - путь до данных для инференса
* `load_state` - путь до весов модели
* `metrics.test_metrics.path_to_save` - путь для сохранения результатов инференса

---

### Инференс продуктов( инструкция для AmazME)
1. Подгрузка данных из РХ
```bash
hdfs dfs -get <PATH TO TABULAR DATA WITH HIDDEN STATES>
```
2. Подгрузка весов моделей из РХ
```bash
hdfs dfs -get /user/team/team_ai_avatar/rusakov/june_pilot/model_weights/cc_model.bin
hdfs dfs -get /user/team/team_ai_avatar/rusakov/june_pilot/model_weights/sa_model.bin
hdfs dfs -get /user/team/team_ai_avatar/rusakov/june_pilot/model_weights/td_model.bin
```
3. Файлы конфигурации
В каждом из файлов нужно указать:
* Путь до данных: `test_dataloader.dataset.path: <PATH TO TABULAR DATA WITH HIDDEN STATES>`
* Путь до модели в зависимости от продукта: `load_state: path_to_<cc/sa/td_model.bin>`
* Путь до сохранения: `metrics.test_metrics.path_to_save: path_to_save/product_name`

4. Запуск инференса
```bash
python -m avatar.cam_inference --config-dir=configs/prom --config-name=cc_inference
python -m avatar.cam_inference --config-dir=configs/prom --config-name=td_inference
python -m avatar.cam_inference --config-dir=configs/prom --config-name=sa_inference
```

