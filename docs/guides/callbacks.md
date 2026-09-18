# Колбэки

Колбэк наблюдает за циклом и может им управлять, но никогда не считает
математику. Всё, что в цикле не является форвардом, бэквардом, шагом
оптимизатора и обвязкой вокруг них — логирование, чекпоинты, ранняя остановка,
EMA, профайлер, метрики производительности — это колбэки.

Устройство самого цикла — в [training.md](training.md).

## Двенадцать событий

| событие | когда | что доступно |
|---|---|---|
| `on_train_begin` | один раз перед первой эпохой | всё, кроме батча |
| `on_epoch_begin` | начало эпохи | `state.epoch` |
| `on_batch_begin` | после переноса на устройство, до форварда | `ctx.batch` |
| `on_forward_end` | после форварда | `ctx.output`, `ctx.loss` |
| `on_backward_end` | градиенты посчитаны, шага ещё не было | градиенты в параметрах |
| `on_optimizer_step` | только на границе накопления | `ctx.grad_norm`, если включена обрезка |
| `on_step_end` | шаг оптимизатора завершён | `state.global_step` увеличен |
| `on_evaluate` | метрики посчитаны | `ctx.metrics` (только на ранге 0) |
| `on_save` | решение сохранять принято | `ctx.checkpoint_dir` |
| `on_epoch_end` | конец эпохи | счётчики эпохи |
| `on_log` | публикуются метрики | `ctx.logs`, `ctx.log_step` |
| `on_train_end` | после последней эпохи | итоговое состояние |

Все хуки получают один и тот же `CallbackContext` и ничего не возвращают.

## Хуки срабатывают на всех рангах

Это самая важная деталь. Цикл **не** фильтрует по рангу: если действие должно
произойти один раз — запись файла, лог в MLflow — проверяйте
`ctx.env.is_main` сами.

Сделано так намеренно: колбэки, которым нужны коллективные операции
(`ThroughputCallback` сводит время по рангам, ранняя остановка участвует в
рассылке решения), должны выполняться везде. Если бы цикл сам ограничивал их
рангом 0, коллективная операция внутри такого колбэка повесила бы прогон.

## Что лежит в контексте

```python
ctx.env             # DistEnv: ранг, размер мира, устройство, коллективы
ctx.state           # TrainerState: эпоха, шаги, число примеров, learning rate
ctx.control         # TrainerControl: флаги, которыми можно управлять циклом
ctx.run_config      # RunConfig: distributed / amp / ddp / compile
ctx.config          # весь конфиг Hydra
ctx.model           # модель, уже РАЗВЁРНУТАЯ из DDP и AveragedModel
ctx.optimizer, ctx.scheduler
ctx.batch, ctx.output, ctx.loss, ctx.grad_norm
ctx.metrics         # результаты валидации (на ранге 0)
ctx.logs, ctx.log_step
ctx.checkpoint_dir
ctx.extra           # общий словарь для обмена между колбэками
ctx.log             # ctx.log({"имя": значение}, step=...)
```

`ctx.model` **всегда развёрнута**: колбэку, который сохраняет веса или читает
параметры, не нужно знать, есть ли DDP. Обёрнутую модель цикл держит при себе.

`ctx.extra` — канал между колбэками. Так `EMACallback` публикует
`ctx.extra["eval_model"]`, и цикл валидируется усреднённой моделью.

## Управление циклом

Единственный способ повлиять на цикл — записать флаг в `ctx.control`:

| флаг | эффект |
|---|---|
| `should_training_stop` | закончить обучение после текущей эпохи |
| `should_epoch_stop` | прервать текущую эпоху |
| `should_evaluate` | провести валидацию на ближайшей границе накопления |
| `should_save` | сохранить чекпоинт (сбрасывается в `True` перед `on_evaluate`) |
| `should_log` | зарезервирован |

`should_training_stop` и `should_save` после `on_evaluate` **рассылаются с
ранга 0** остальным. Поэтому решение, принятое на другом ранге, будет
проигнорировано: принимать его должен тот, кто видел метрики.

## Порядок

Колбэки вызываются в порядке списка. Значимо это ровно в одном месте: **колбэк
ранней остановки должен идти раньше колбэка чекпоинтов**, потому что первый
снимает флаг, который читает второй.

Стандартный список (когда `callbacks:` в конфиге нет) собирается так:

```
MLflowCallback, ProgressBarCallback, TrainStatsCallback,
[EMACallback], [TrainMetricsCallback],
[PerfMetricsCallback, ThroughputCallback], [ProfilerCallback],
[EarlyStoppingCallback], CheckpointCallback
```

Квадратные скобки — включается по конфигу. Явный список `callbacks:` заменяет
этот набор **целиком**, поэтому не забудьте про чекпоинты.

## Как написать свой колбэк

```python
from fmlib.train import TrainerCallback
from fmlib.train.state import CallbackContext


class GradientNormAlarm(TrainerCallback):
    """Предупредить, если норма градиента превысила порог."""

    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold
        self.exceeded = 0

    def on_optimizer_step(self, ctx: CallbackContext) -> None:
        if ctx.grad_norm is None:
            return                      # обрезка выключена — нормы нет
        if float(ctx.grad_norm) > self.threshold:
            self.exceeded += 1

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        total = ctx.env.all_reduce_sum(float(self.exceeded))
        if ctx.env.is_main and ctx.log is not None:
            ctx.log({"grad/exceeded_steps": total}, step=ctx.state.epoch)
        self.exceeded = 0
```

Подключение — обычным `_target_`:

```yaml
callbacks:
  - _target_: fmlib.train.MLflowCallback
  - _target_: fmlib.train.ProgressBarCallback
  - _target_: mypackage.callbacks.GradientNormAlarm
    threshold: 50.0
  - _target_: fmlib.train.CheckpointCallback
    checkpoint_dir: best_models/my_experiment/my_run
```

Что стоит соблюдать:

* **Наследуйтесь от `TrainerCallback`** и реализуйте только нужные хуки —
  остальные пустые.
* **Коллективные операции вызывайте на всех рангах.** Обратите внимание на
  пример: `all_reduce_sum` — вне проверки `is_main`, а лог — внутри. Обратный
  порядок повесил бы прогон.
* **Логируйте через `ctx.log`**, а не напрямую в MLflow: так значения пройдут
  через все установленные логирующие колбэки.
* **Не трогайте `ctx.batch` и `ctx.output`.** Цикл использует их дальше.
* **Обнуляйте своё состояние** на границе эпохи, если оно поэпохное.
* Держать в колбэке тензоры с графом — верный способ утечки памяти:
  `.detach()` обязателен.

## Готовые колбэки

| класс | что делает |
|---|---|
| `MLflowCallback` | владеет прогоном MLflow на ранге 0, пишет параметры и метрики |
| `ProgressBarCallback` | прогресс-бар; продвигается из `on_batch_begin` и не оборачивает даталоадер |
| `CheckpointCallback` | пишет чекпоинт и веса, ротирует старые |
| `EarlyStoppingCallback` | останавливает обучение и запрещает сохранение |
| `EMACallback` | усредняет веса, публикует `ctx.extra["eval_model"]` |
| `TrainStatsCallback` | средние по эпохе: потери, learning rate, норма градиента |
| `TrainMetricsCallback` | считает метрики на обучающей выборке |
| `PerfMetricsCallback` | системные и шардовые метрики — см. [../best_practices/performance_metrics.md](../best_practices/performance_metrics.md) |
| `ThroughputCallback` | пропускная способность и перекос между рангами |
| `ProfilerCallback` | `torch.profiler` |
