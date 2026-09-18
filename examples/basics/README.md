## Базовые операции с пайплайном
### Клонирование репозитория + установка зависимостей
```bash
git clone git@github.com:v4ndi/substrate.git avatar_fm
cd avatar_fm
python -m pip install -e ".[spark,dev]"
```

Подробнее про extras (`spark`, `catboost`, `dev`) — в корневом `README.md`.

### Данные

Датасеты читают parquet как с локального диска, так и напрямую с HDFS — достаточно
указать URI в `path`:

```yaml
train_dataloader:
  dataset:
    _target_: fmlib.data.TabularDataset
    path: hdfs://arnsdpsbx/user/team/team_ai_avatar/avatar_fm/examples/campaign_demo
```

Требуется установленный клиент libhdfs: `HADOOP_HOME` (или `ARROW_LIBHDFS_DIR`) и
`CLASSPATH` должны быть в окружении. Если нужны явные параметры подключения,
добавьте блок `filesystem`:

```yaml
    filesystem: {type: hdfs, host: arnsdpsbx, port: 8020, user: team}
```

Каждый DataLoader-воркер поднимает свою JVM, поэтому при чтении с HDFS следите за
`num_workers`.

Альтернатива — скопировать данные на локальный диск заранее:

```bash
cd <root_dir>/avatar_fm/examples/demo_campaign
mkdir data && cd data
hdfs dfs -get hdfs://arnsdpsbx/user/team/team_ai_avatar/avatar_fm/examples/campaign_demo
```

---

### Навигация
* `artifacts` - вспомогательные файлы, конфиги препроцессоров
* `tabular_preprocessing.ipynb` - Предобработка табличных признаков + join скрытых состояний
* `tabular_dataset.ipynb` - Инициализация табличного датасета и работа с ними
* `metrics.ipynb` - Пример работы с базовым классом для расчёта метрик
  
---

