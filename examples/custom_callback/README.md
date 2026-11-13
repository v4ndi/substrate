# Свой колбэк

Минимальный рабочий пример точки расширения «колбэк»: `callback.py` содержит
`GradientNormAlarm` — колбэк, который считает, как часто норма градиента
превышала порог, и раз в эпоху логирует долю таких шагов.

Контракт и список событий — в
[../../docs/guides/callbacks.md](../../docs/guides/callbacks.md).

## Подключение

```yaml
callbacks:
  - _target_: avatar.train.MLflowCallback
  - _target_: avatar.train.ProgressBarCallback
  - _target_: examples.custom_callback.callback.GradientNormAlarm
    threshold: 1.0
  - _target_: avatar.train.CheckpointCallback
    checkpoint_dir: best_models/my_experiment/my_run
```

Явный список `callbacks:` **заменяет стандартный целиком**, поэтому колбэк
чекпоинтов нужно перечислить самому. Если он не нужен, а нужен только свой
колбэк вдобавок к стандартным — проще собрать список из
`avatar.train.build_default_callbacks` в Python, чем перечислять всё в YAML.

## Две вещи, которые легко сделать неправильно

**Хуки срабатывают на всех рангах.** Цикл не фильтрует по рангу. Всё, что должно
случиться один раз, проверяет `ctx.env.is_main` само.

**Коллективная операция вызывается вне этой проверки.** В примере
`all_reduce_sum` идёт до `if ctx.env.is_main`, а логирование — после. Обратный
порядок означал бы, что в коллективную операцию зайдёт только ранг 0, а
остальные до неё не дойдут — прогон повиснет до таймаута.

## Про `ctx.grad_norm`

Он заполняется, только если задан `train.clip_grad_norm`: норму считает именно
обрезка. Без неё колбэк молчит — это проверяется тестом.

## Проверка

```bash
python -m pytest tests/examples
```

Тесты создают колбэк ровно тем же способом, что и конфиг — через Hydra
`_target_`, — поэтому переименование в `avatar.train` ломает тест, а не читателя.
