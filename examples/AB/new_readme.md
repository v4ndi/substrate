# Установка пайплайна

Инструкцию по установке зависимостей можно найти в README репозитория

ИНСТРУКЦИЯ НАПИСАНА ДЛЯ ЗАДАЧИ AB БЕЗ AB, ДЛЯ КЛЮЧЕВЫХ СОБЫТИЙ ВСЕ АНАЛОГИЧНО, ТОЛЬКО КОНФИГИ В `avatar_fm/experiments/key_events/configs/`

# Оглавление

1. [Получение данных](#получение-данных)  
2. [Обучение модели Core FM](#обучение-модели-core-fm)  
3. [Подготовка модели для финального обучения](#подготовка-модели-для-финального-обучения)  
4. [Финальное обучение](#финальное-обучение)  
5. [Выбор лучшей модели](#выбор-лучшей-модели)  
6. [Калибровка модели](#калибровка-модели)  
7. [Инференс](#инференс)  

# Получение данных

- Данные собираются с использованием Spark.  
- Даты разбиения:  
  - Train: 2023.01.01 - 2024.11.31  
  - Validation: 2024.12.01 - 2024.12.31  
  - Test: 2025.01.01 - 2025.03.31  
- Одновременно формируется словарь, необходимый для расшифровки предсказанных ВНВ при инференсе.  
- Ноутбук для сбора данных:  
  `avatar_fm/experiments/ab/spark/seq_collect_new.ipynb`
- Пример данных
    - Train + val `/arnsdpsbx/user/team/team_ai_avatar/hive/datasets/business_compare/abab/ds/new_train_chains_q1/chains`
    - Test `/arnsdpsbx/user/team/team_ai_avatar/hive/datasets/business_compare/abab/ds/chains_infer`
- Пример данных для задачи предсказания ключевых событий
    - Train - `/arnsdpsbx/user/team/team_ai_avatar/ds/zorin/key_events/old_targets_1_5_year_full_train_mega_repartited_right`
    - Val - `/arnsdpsbx/user/team/team_ai_avatar/ds/zorin/key_events/old_targets_1_5_year_full_val_mega_right`
    - Test - `/arnsdpsbx/user/team/team_ai_avatar/ds/zorin/key_events/old_targets_1_5_year_full_test_mega_right`

# Обучение модели Core FM

1. Обучается модель Core FM для обработки транзакционных данных.  
2. Пример конфига:  
   `avatar_fm/experiments/ab/configs/core_model/core_training_ab.yaml`  
   *(обязательно укажите пути до ваших данных)*  
3. Запуск обучения:  

`accelerate launch -m avatar.train --config-dir 'путь_до_конфига' --config-name 'название_конфига'`

4. Просмотр метрик в терминале:  

`mlflow ui --port 5000 --host 127.0.0.1 --default-artifact-root ./mlruns`

- Для доступа откройте в браузере:  
`https://jupyterhub-datalab.apps.prom-datalab.ca.sbrf.ru/user/логин_omega-sbrf-ru/proxy/5000/#/`

# Подготовка модели для финального обучения

- После обучения в папке `best_models` ищите нужный эксперимент и модель (файл `model.bin`), соответствующие желаемой эпохе.  
- Для дальнейшего обучения удалите классификационный слой с помощью ноутбука:  
`avatar_fm/notebooks/cut_a_head.ipynb`

# Финальное обучение

1. Используйте пример конфига:  
`avatar_fm/experiments/ab/configs/final_training/ab_example.yaml`  
*(укажите пути к данным и веса модели из предыдущего шага в параметре `encoder_weights`)*  
2. Запуск аналогичен разделу обучения Core FM.

# Выбор лучшей модели

- Оценивайте модели по метрикам на валидационном наборе в mlflow и выбирайте оптимальную. Рекомендовано использовать лучшую по val/T-map модель, т.к. она устойчива к трешхолдам, поэтому можно получить лучшую метрику после калибровки трешхолдов. 

# Калибровка модели

- Выполните калибровку для улучшения качества предсказаний.  
- Пример конфига:  
`avatar_fm/experiments/ab/configs/calibrate/ab_calib.yaml`  
- В конфиге укажите:  
- Путь к калибровочному датасету (можно использовать валидационный, но рекомендуется взять 1/3 валидации для ускорения)  
- `path_to_save_thresholds` — путь для сохранения порогов  
- `load_state` — модель для калибровки  


- Для автоматической генерации конфига калибровки рекомендуется использовать файл `avatar_fm/experiments/ab/configs/make_calib.py`
- Запуск: 
`python make_calib.py путь_до_train_config --root_dir=путь_до_корневой_директории_в_которой_best_models --calib_folder=папка_в_которую_сохраняются_калибровочные_трешхолды`

- Запуск калибровки:  
`accelerate launch -m avatar.inference --config-dir 'путь_до_конфига' --config-name 'название_конфига'`

# Инференс

- Пример конфига:  
`avatar_fm/experiments/ab/configs/inference/march_inference.yaml`  
- Для каждого месяца создавайте отдельный конфиг с корректным `target_date` (последний день месяца) и путь сохранения результатов.  
- В конфиге укажите:  
- Путь к словарю, полученному на Spark  
- Опционально путь к порогам калибровки (`path_to_thresholds`), если применима калибровка  
- Путь для словаря перевода в обычный формат `vnv_map_dict` или `key_events_dict` в случае ключевых событий (примеры словарей находятся в `avatar_fm/experiments/ab/` и `avatar_fm/experiments/key_events/` соответственно)
- Запуск аналогичен разделу калибровки:

`accelerate launch -m avatar.inference --config-dir 'путь_до_конфига' --config-name 'название_конфига'`

# СИТУАТИВНО. Инференс и калибровка используя мат. ожидания 

- Если требуется предсказать количество каждого продукта за месяц, то есть смысл получать предсказания не цепочек а мат.ожиданий количества каждого продукта
- Сначала нужно обучить линейные регрессии, которые являются калибровочными моделями для мат. ожиданий. Пример конфига - `avatar_fm/experiments/ab/configs/math_expectations_calibrate/me_calib_example.yaml`
Обязательно нужно прописать пути до своих данных и указать путь сохранения моделей `me_models_path`
Запуск аналогично обычной калибровке 
`accelerate launch -m avatar.inference --config-dir 'путь_до_конфига' --config-name 'название_конфига'`
- После этого используется обычный инференс, в котором нужно прописать `path_to_me_models`. Пример конфига `avatar_fm/experiments/ab/configs/inference/me_inference_example.yaml`

# APPENDIX по периспользованию в задачах без фиксированного горизнта 
- В пайплайне AB всегда предсказывается все события следующего календарного месяца и нужно предсказывать с начала месяца, поэтому в декодер передается hidden state с последнего события предыдущего месяце (`outputs.last_hidden_state[:,-1,:]`). Если события у нас однородные (модель можно обучать с любого элемента последовательности, не потеряв целостность), то лучше передавать сразу `outputs.last_hidden_state` из энкодера в декодер целиком, используя в DetectionLoss параметр `use_more_states=True`. (эта часть реализована в модуле `avatar/pipeline/fixed_horizon.py`)