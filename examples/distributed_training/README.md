# Распределённое обучение

Запускаемый пример multi-GPU и multi-node обучения на синтетических данных.
Ничего выгружать не нужно: данные генерирует скрипт.

## Запуск

```bash
python examples/distributed_training/generate_data.py
```

Скрипт пишет 40 000 обучающих и 8 000 валидационных записей и печатает готовую
команду запуска — с абсолютным путём до данных. Она выглядит так:

```bash
torchrun --standalone --nproc_per_node=2 -m avatar.train \
    --config-dir=examples/distributed_training/configs --config-name=synthetic \
    data_dir=/абсолютный/путь/examples/distributed_training/data
```

За три эпохи функция потерь падает примерно с 0.50 до 0.16, а
`valid_roc_auc_score` доходит до ~0.99: цель — зашумлённая линейная функция
признаков, то есть задача заведомо решаемая. Это и нужно от примера: если у вас
получилось иначе, дело в окружении, а не в данных.

## Один процесс, несколько карт, несколько узлов

Один и тот же конфиг работает во всех трёх случаях — меняется только команда:

```bash
# один процесс, без лаунчера
python -m avatar.train --config-dir=configs --config-name=synthetic data_dir=...

# все карты одной машины
torchrun --standalone --nproc_per_node=8 -m avatar.train ...

# два узла по 8 карт: то же самое на каждом узле, меняется только --node_rank
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=head-node:29500 \
    -m avatar.train ...
```

Кода это не касается: без `torchrun` в окружении нет `WORLD_SIZE`, поэтому
размер мира равен 1, группа процессов не создаётся, а модель не оборачивается в
DDP.

## Проверить на машине без GPU

Пример полностью работает на CPU — так его и проверяли:

```bash
CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=2 -m avatar.train ...
```

Без видимых карт бэкенд сам выбирается как gloo. Два ранга NCCL не могут делить
одно устройство, два ранга gloo — могут, поэтому это единственный способ
проверить многоранговое поведение на одной карте.

## Что гарантирует шардирование

`DistributedSampler` здесь нет и быть не должно: `TabularDataset` делит **записи**
между рангами и воркерами сам. При `drop_tail: True` (по умолчанию) каждый ранг
получает ровно `total // world_size` записей, и это то, что гарантирует
одинаковое число батчей на всех рангах.

Если ранги разойдутся в числе батчей, один из них уйдёт в коллективную операцию,
до которой остальные не дойдут, и прогон повиснет до таймаута. Подробнее — в
[../../docs/guides/datasets.md](../../docs/guides/datasets.md).

## Что смотреть в конфиге

```yaml
distributed:
  backend: null                    # null -> nccl на GPU, gloo иначе
  gradient_accumulation_steps: 1
amp: no                            # no | fp16 | bf16
ddp:
  find_unused_parameters: False
```

* `gradient_accumulation_steps` увеличивает эффективный батч
  (`batch_size * world_size * N`), а не экономит память при том же батче.
* `amp: bf16` на Ampere и новее — обычно лучший вариант: та же экономия памяти,
  что у fp16, без риска переполнения.
* `find_unused_parameters: True` нужен, только если DDP ругается на
  неиспользованные параметры: он стоит заметных накладных расходов.

Полный разбор — в
[../../docs/configuration/distributed.md](../../docs/configuration/distributed.md).

## Куда всё складывается

Каталоги создаются относительно места запуска:

* `best_models/distributed_training_example/debug/{шаг}/` — чекпоинты; при
  `max_saved_checkpoints: 2` остаются два последних;
* `mlruns/` — метрики MLflow.

`run_name: debug` в конфиге стоит намеренно: любое другое имя запрещает повторный
запуск в уже существующий каталог, чтобы не затереть чужой прогон.
