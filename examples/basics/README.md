## Базовые операции с пайплайном
### Клонирование репозитория + установка зависимостей
```bash
git clone https://df-bitbucket.ca.sbrf.ru/scm/dsrpt/avatar_fm.git
cd avatar_fm
chmod +x install.sh
./install.sh
```

### Загрузка данных
```bash
cd <root_dir>/avatar_fm/examples/demo_campaign
mkdir data && cd data
hdfs dfs -get hdfs://arnsdpsbx/user/team/team_ai_avatar/avatar_fm/examples/campaign_demo
```

---

### Навигация
* `atrifacts` - вспомогательные файлы, конфиги препроцессоров
* `tabular_preprocessing.ipynb` - Предобработка табличных признаков + join скрытых состояний
* `tabular_dataset.ipynb` - Инициализация табличного датасета и работа с ними
* `metircs.ipynb` - Пример работы с базовым классом для рассчета метрик
  
---

