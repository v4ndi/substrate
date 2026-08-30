# Метрики производительности

Подсистема, которую невозможно найти, не читая исходники: блок `logging:`
читается тренером, но не встречается **ни в одном** конфиге репозитория.
Между тем именно она отвечает на вопросы «почему добавление карт не ускоряет
обучение» и «мы упираемся в GPU или в чтение данных».

## Включение

```yaml
logging:
  enable: True
  enable_profiler: False
  performance_metrics:
    enabled: True
    sampling_interval_sec: 2.0
    native_system_metrics: True
```

| ключ | по умолчанию | что делает |
|---|---|---|
| `enable` | `True` | общий выключатель логов, прогресс-бара и MLflow |
| `enable_profiler` | `False` | включает `torch.profiler` с записью трасс |
| `performance_metrics.enabled` | `False` | всё описанное ниже |
| `performance_metrics.sampling_interval_sec` | `2.0` | как часто снимать системные счётчики |
| `performance_metrics.native_system_metrics` | `True` | использовать встроенный сбор системных метрик MLflow |

## Чего это стоит

Сведения выполняются **раз в эпоху**, а не на каждом шаге, поэтому в
горячем цикле остаётся только отметка времени на батч и опрос счётчиков не чаще
`sampling_interval_sec`. На типичных прогонах накладные расходы неразличимы, и
включать это можно постоянно.

Исключение — `enable_profiler`. Профайлер записывает трассы и заметно замедляет
обучение; включайте его точечно.

## Пропускная способность

| метрика | что означает |
|---|---|
| `throughput/samples_per_sec` | суммарно по всем рангам |
| `throughput/samples_per_sec_per_rank` | то же, поделённое на размер мира |
| `throughput/step_time_sec` | среднее время шага по самому медленному рангу |
| `throughput/step_time_imbalance_pct` | разрыв между самым медленным и самым быстрым рангом |
| `throughput/non_compute_sec` | время эпохи, не занятое вычислениями |

**`step_time_imbalance_pct` — главное число при масштабировании.** Это разрыв
между рангами в процентах от самого медленного, и именно он превращается в
простаивающий GPU на all-reduce градиентов: быстрые ранги ждут медленный.
Значение в единицы процентов нормально, десятки — повод искать перекос в
данных.

`samples_per_sec_per_rank`, падающий при добавлении рангов, — это и есть
плохая масштабируемость; сравните его с `step_time_imbalance_pct`, чтобы
понять, дело в перекосе или в коммуникации.

## Эпоха

| метрика | что означает |
|---|---|
| `train_epoch/wall_sec` | длительность эпохи по самому медленному рангу |
| `train_epoch/global_samples` | сколько записей обработано всеми рангами |
| `train_epoch/global_samples_per_sec` | суммарная пропускная способность |
| `train_epoch/between_epochs_sec` | время между концом эпохи и началом следующей |
| `startup/before_first_epoch_sec` | всё, что произошло до первой эпохи |
| `startup/first_batch_sec` | ожидание первого батча первой эпохи |

`between_epochs_sec` и `first_batch_sec` — про даталоадер. Большое время до
первого батча означает, что воркеры долго стартуют или сканирование не
закэшировалось.

## Шардирование

Разовые метрики, снимаемые после сканирования:

| метрика | что означает |
|---|---|
| `shard/scan_sec` | сколько заняло сканирование |
| `shard/scan_rows_per_sec` | скорость сканирования |
| `shard/total_raw_rows` | строк в корпусе |
| `shard/total_valid_rows` | строк после фильтра |
| `shard/dropped_tail_rows` | отброшено хвостом |
| `shard/parquet_file_count` | число файлов |
| `shard/filter_cache_hit` | попал ли кэш фильтров |
| `shard/scan_rank_count` | сколько рангов реально сканировало |

Поэпохные:

| метрика | что означает |
|---|---|
| `shard_epoch/row_read_amplification` | во сколько раз прочитано строк больше, чем понадобилось |
| `shard_epoch/io_read_amplification` | то же по байтам |
| `shard_epoch/filter_time_pct` | доля времени на фильтрацию |
| `shard_epoch/record_count_delta` | разброс числа записей между рангами |
| `shard_epoch/batch_count_delta` | разброс числа батчей между рангами |

**`batch_count_delta` обязан быть нулём.** Ненулевое значение означает, что
ранги разошлись в числе батчей — прогон рано или поздно повиснет на
коллективной операции. Это самая полезная метрика во всём наборе: она ловит
нарушение паритета сканирования и итерации до того, как оно проявится
зависанием.

`row_read_amplification` заметно больше единицы означает, что фильтр отбрасывает
много уже прочитанного — есть смысл посмотреть на `read_columns` или на
партиционирование данных.

## Система

| метрика | что означает |
|---|---|
| `system_epoch/cpu_utilization_pct_mean` / `_max` | загрузка CPU |
| `system_epoch/memory_utilization_pct_max` | пик памяти хоста |
| `system_epoch/gpu_utilization_pct_mean` / `_min` | загрузка GPU |
| `system_epoch/gpu_memory_utilization_pct_max` | пик памяти GPU |
| `system_epoch/disk_read_mib_per_sec` | чтение с диска |
| `system_epoch/network_receive_mib_per_sec` / `_transmit_` | сеть |
| `system_epoch/host_sampler_count` / `gpu_sampler_count` | сколько раз успели снять счётчики |

`gpu_utilization_pct_mean` заметно ниже 100 при высоком
`disk_read_mib_per_sec` — классическая картина «упёрлись в данные»:
увеличивайте `num_workers` или сокращайте `read_columns`.

Счётчики снимает только локальный главный процесс каждого узла, чтобы не
опрашивать одни и те же системные счётчики восемь раз.

## Что записывается как параметры прогона

При включённых метриках производительности в параметры MLflow дополнительно
уезжает конфигурация запуска: `shard`, `drop_tail`, `rotate_tail`,
`filter_cache`, `scan_workers_per_rank`, `batch_size_per_rank`,
`num_workers_per_rank`, `parquet_file_count`, `dataset_size_bytes`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `torch_num_threads`,
`available_cpu_cores`.

Это позволяет сравнивать прогоны между собой, не восстанавливая по памяти, с
какими настройками каждый из них запускался.

## Профайлер

```yaml
logging:
  enable_profiler: True
```

Включает `torch.profiler` по расписанию «пропустить 3, прогреть 1, писать 5,
повторить дважды». Трассы сохраняются в
`/home/datalab/nfs/profile_traces/{experiment_name}/{run_name}/{время}/trace.json`
и открываются в `chrome://tracing` или Perfetto.

Путь сохранения зашит в код (`avatar.utils.init_modules.init_profiler`) — если
он вам не подходит, это место придётся править.
