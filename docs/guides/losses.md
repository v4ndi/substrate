# Функции потерь

Функция потерь — отдельный `nn.Module`, которым владеет пайплайн. Не backbone и
не тренер:

```
тренер ──вызывает──► pipeline.forward(batch) ──► output (есть .loss)
                          │
                          ├─ backbone(...)     # avatar/nn — потерь не считает
                          └─ self.loss(...)    # avatar/losses
```

Такое разделение решает две задачи. Backbone без потерь можно прогнать на
инференсе, где таргетов нет. Тренер без потерь ничего не знает о семантике
таргетов конкретной задачи — а их у задач много разных.

## Контракт

Один-единственный: возвращать `LossOutput`.

```python
@dataclass
class LossOutput:
    loss: torch.Tensor | None = None
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    num_items: dict[str, torch.Tensor] | None = None
```

Список аргументов контрактом **не** является: он остаётся делом пайплайна,
поскольку только пайплайн знает, что значат его таргеты. `ClassificationLoss`
принимает `(logits, targets, model)`, `NextKTokensLoss` — список горизонтов.

| поле | когда заполнять |
|---|---|
| `loss` | скаляр, по которому вызовут `backward()` |
| `components` | отдельные слагаемые — для логирования |
| `num_items` | число валидных элементов на голову; см. ниже |

## Когда нужен `num_items`

Для многоголовых потерь — таких, где у каждой головы своё число валидных
элементов. Например, при предсказании следующих K событий каждая голова видит
разное количество непаддинговых позиций.

Наличие `num_items` — это **сигнал тренеру взвешивать потери по токенам между
рангами**:

```
loss_head = loss_head_local * world_size / сумма_по_рангам(num_items_head)
```

Множитель `world_size` компенсирует то, что DDP усредняет градиенты по рангам;
после компенсации получается честная взвешенная по токенам сумма. Без этого
ранг, которому достались более короткие последовательности, весил бы на токен
больше соседей.

В таком случае `loss` оставляется `None`: скаляр соберёт тренер
(`avatar.train.loss_reduce.calculate_output_loss`).

Если у всех голов одинаковое число элементов или голова одна, `num_items` не
нужен — заполняйте `loss`, а `components` используйте для логов.

## Подмена из конфига

У пайплайнов есть ключ `loss:`. По умолчанию пайплайн строит функцию потерь
сам, но её можно заменить, не трогая код:

```yaml
model:
  _target_: avatar.pipeline.tabular.TabularClassification
  num_classes: 2
  loss:
    _target_: avatar.losses.ClassificationLoss
    num_classes: 2
    task_type: classification
    l1_weight: 0.01
```

У `ClassificationLoss` есть ещё одна точка подмены — `loss_fn`: она заменяет
критерий, оставляя обвязку с L1-регуляризацией:

```yaml
  loss:
    _target_: avatar.losses.ClassificationLoss
    loss_fn:
      _target_: torch.nn.BCEWithLogitsLoss
      pos_weight: [3.0]
```

## Композиция

`CompositeLoss` — взвешенная сумма именованных потерь:

```yaml
loss:
  _target_: avatar.losses.CompositeLoss
  losses:
    task:
      _target_: avatar.losses.ClassificationLoss
      num_classes: 2
    kld:
      _target_: avatar.losses.KLDLoss
      alpha: 0.5
  weights: {task: 1.0, kld: 0.1}
```

Все вложенные потери получают **одни и те же аргументы**, поэтому они должны
быть согласованы по сигнатуре. Слагаемые каждой из них попадают в результат под
её именем (`kld/что-то`), так что композиция композиций всё равно даёт плоский
словарь компонент.

Вес для неизвестного имени — ошибка (`ValueError`), а не молчаливое
игнорирование: опечатка в имени иначе означала бы, что вес просто не применился.

## Как написать свою

```python
import torch
import torch.nn as nn

from avatar.losses import Loss, LossOutput


class FocalLoss(Loss):
    """Focal loss: занижает вклад легко классифицируемых примеров."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets, model=None) -> LossOutput:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        focal = ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()
        return LossOutput(loss=focal, components={"ce": ce.mean()})
```

```yaml
model:
  _target_: avatar.pipeline.tabular.TabularClassification
  num_classes: 2
  loss:
    _target_: mypackage.losses.FocalLoss
    gamma: 2.0
```

Что стоит соблюдать:

* **Наследуйтесь от `Loss`** и возвращайте `LossOutput`. Голый тензор сломает
  пайплайн, который ожидает `.loss` и `.components`.
* **Повторите сигнатуру той функции потерь, которую заменяете.** Пайплайн
  вызывает её конкретным образом: `ClassificationLoss` — как
  `self.loss(logits, targets, model=self.encoder)`.
* **Принимайте `model=None`**, даже если он не нужен: так функцию можно
  подставить туда, где пайплайн его передаёт.
* `components` — для наблюдаемости. Их подхватит `MultiLossMetric` или
  `UniversalLossesMetric`.

## Готовые функции потерь

| класс | что считает |
|---|---|
| `ClassificationLoss` | MSE / BCE / CrossEntropy по `(num_classes, task_type)`, плюс L1 |
| `NextKTokensLoss` | потери по головам и горизонтам для предсказания следующих K событий |
| `CompositeLoss` | взвешенная сумма именованных потерь |
| `L1RegularizationLoss` | L1 по параметрам, чьё имя содержит подстроку |
| `DirectUpliftLoss` | прямая оптимизация uplift |
| `KLDLoss`, `ContrastiveLoss`, `KLDxContrastiveLoss`, `KLDxContrastiveGridLoss` | исследовательские |
| `ResearchLosses` | выбирает одну из перечисленных выше по имени |
| `GoldFishLoss` | см. докстринг класса |

`build_task_loss_fn(num_classes, task_type)` — вспомогательная функция,
выбирающая критерий: `num_classes=1` + `regression` → MSE, `num_classes=1` +
`classification` → BCEWithLogits, `num_classes>1` + `classification` →
CrossEntropy. Прочие сочетания — `ValueError`.

## Особый случай: `NextKTokensLoss`

Головы предсказания живут на пайплайне `NextKTokensPrediction`, а не на функции
потерь. Причина не архитектурная, а совместимости: головы содержат параметры, и
их перенос переименовал бы все ключи `lm_heads.*` в существующих чекпоинтах.

В модуле потерь остаются сдвиг меток, критерии по типам признаков и взвешивание
по горизонтам: `loss_k = loss_k / k ** horizion_loss_weight`.
