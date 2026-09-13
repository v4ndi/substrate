# Границы: `avatar/pipeline`, `avatar/losses`, `avatar/nn`

Статус: **принято** (2026-09-13). Сокращает `avatar/pipeline` до двух семейств
и фиксирует, кто за что отвечает на границе между тремя пакетами.

Повод: в `avatar/pipeline` лежали четыре семейства, два из которых —
`sequence` и `multi_task` — остались без единого рабочего конфига и без
примера. Заодно понадобилось записать, зачем вообще существует
`avatar/losses`, раз половина его содержимого не следует собственному
контракту.

---

## 0. Что удалено

| путь | почему |
|---|---|
| `avatar/pipeline/sequence/` | `NextKTokensPrediction`, `SequenceModelWithAggregation`, `SequenceClassification` |
| `avatar/pipeline/multi_task/` | `MMoE`, `PLE`, `MultiTaskResponse`, `MultiTaskUplift` и их backbone-ы |
| `avatar/losses/next_k_tokens.py` | потребитель был один — `NextKTokensPrediction` |
| `avatar/losses/multi_task.py` | не экспортировался из `__init__` и не упоминался нигде ещё до этого решения |
| `avatar/losses/gold_fish.py` | ни конфига, ни теста, ни вызова; осмыслен только для next-token |
| `avatar/losses/direct_uplift_loss.py` | не конструируется, см. §3 |
| `examples/next_event_prediction/` | единственный пример на удалённые пайплайны |
| `experiments/.../prom/sequence/inference_sequence.yaml` | единственный конфиг на `SequenceModelWithAggregation` |
| dataclass-ы `SequenceOutput`, `MMoEOutput`, `MMoeUpliftOutput`, `MultiTaskxGroup*Output`, `BaseClassificationOutput` | у первых четырёх ушли производители; последний не имел их и раньше |

**Что осталось нетронутым:** `avatar/data/sequential` (препроцессинг, датасет,
`EventSequenceBatch`), `avatar/nn/sequential` (`EventEncoder`,
`TransformersWrapper`), `BaseSequenceOutput`. Удалён слой задачи, а не работа с
последовательностями. Вернуть постановку — значит написать пайплайн, а не
восстанавливать препроцессинг.

**Что осиротело и оставлено сознательно:**

* `avatar.metrics.MultiLossMetric` требует `outputs.losses` и
  `outputs.num_items`. Такие поля были только у `SequenceOutput`. Метрика
  рабочая и покрыта тестами, но производителя у неё сейчас нет.
* `avatar.metrics.CollectEmbeddings` требует `aggregated_hidden_state` —
  поле, которое отдавал `SequenceModelWithAggregation`.
* `LossOutput.num_items` и ветка взвешивания по токенам в
  `avatar.train.loss_reduce.calculate_output_loss` — механизм живой, но
  единственная функция потерь, которая его заполняла, удалена.

Это не мёртвый код в смысле «можно удалить не думая»: это рабочие механизмы,
оставшиеся без текущего пользователя. Удалять их — отдельное решение, которое
здесь не принимается.

---

## 1. Правила слоёв

Четыре слоя, и к каждому — проверка, по которой видно нарушение.

### `avatar/nn` — представление

Вход → представление. Не знает ни таргетов, ни задачи, ни потерь.

> **Проверка:** блок можно прогнать на инференсе, где колонки таргета в данных
> физически нет.

Сегодня выполняется: под `avatar/nn` нет ни одного вызова функции потерь, и
три `__init__` это прямо декларируют.

### `avatar/pipeline` — задача

Единственный слой, который знает семантику: какая колонка батча — таргет, что
означает ширина головы, какие аргументы существуют у функции потерь. Он же —
композиция: это он собирает `embedding + encoder + aggregation + head`.

> **Проверка:** имя пайплайна стоит в `model:`, и ничто под `avatar/nn` о нём
> не знает.

### `avatar/losses` — критерий

Скаляр, по которому пойдёт `backward()`, плюс слагаемые для логов.

> **Проверка:** функцию потерь можно заменить из конфига, не трогая код
> пайплайна.

### `avatar/train` — цикл

Читает у выхода `loss` (или `losses` + `num_items`) и больше ничего.

> **Проверка:** добавление новой задачи не требует правок в `avatar/train`.

---

## 2. Нужен ли `avatar/losses`?

**Нужен — но по одной причине, и она узкая.**

Критерий — это `(logits, targets) -> скаляр`. Для такого torch самодостаточен,
и отдельный пакет был бы просто лишним слоем над `nn.BCEWithLogitsLoss`.

Пакет оправдан ровно тем, что у нас есть потери, которым `(logits, targets)`
**не хватает** — им нужно что-то ещё изнутри модели:

| чего требует сверх логитов и таргетов | кто | почему torch так не умеет |
|---|---|---|
| сам модуль, чтобы пройти по `named_parameters()` | `L1RegularizationLoss` внутри `ClassificationLoss` | штраф берётся по именам параметров, а не по выходу |
| флаг воздействия и агрегированное представление | `KLDLoss`, `ContrastiveLoss` | слагаемое считается **между** группами воздействия, а не по записи |
| число валидных элементов на голову | контракт `num_items` | взвешивание по токенам между рангами; сейчас без пользователя |

Три пункта — это и есть содержательное основание пакета. Всё остальное, что в
нём лежало, либо оборачивало torch без добавленной стоимости, либо не
работало.

Отсюда следствие, которое стоит держать в голове при следующем добавлении:
**если новая функция потерь принимает только логиты и таргеты, ей место не в
`avatar/losses`, а в конфиге как `torch.nn.*`** — механизм подмены `loss_fn:`
у `ClassificationLoss` для этого и существует.

---

## 3. Где код расходится с декларацией

Всё ниже — не гипотезы: проверено запуском.

### 3.1. Протоколов не один, а три

`docs/guides/losses.md` утверждает: «Один-единственный [контракт]: возвращать
`LossOutput`». Из трёх оставшихся пайплайнов ему следует один.

| пайплайн | ключ конфига | как вызывает | что ожидает получить |
|---|---|---|---|
| `TabularClassification` | `loss:` | `self.loss(logits, targets, model=self.encoder)` | `LossOutput` |
| `SLearner` | `loss_fn:` | по `isinstance` — либо `(logits, targets)`, либо `(logits=, is_treat=, dist=, targets=)` | голый тензор |
| `SupervisedLearner` | `loss_fn:` | `self.loss_fn(logits.squeeze(1), targets.float())` | голый тензор |

Три пайплайна — три способа позвать функцию потерь и два разных имени ключа.

### 3.2. Диспетчеризация по типу в `SLearner`

`avatar/pipeline/uplift/s_learner.py:241`:

```python
if isinstance(self.loss_fn, nn.CrossEntropyLoss):
    loss = self.loss_fn(logits, targets)
else:
    loss = self.loss_fn(logits=..., is_treat=..., dist=..., targets=...)
```

«Обычная функция потерь» опознаётся по тому, что она **в точности**
`nn.CrossEntropyLoss`. Подставить `nn.BCEWithLogitsLoss` или любого наследника
`Loss` нельзя: их позовут с четырьмя именованными аргументами. Это же и
причина, по которой `ContrastiveLoss` и `KLDLoss` не наследуют `Loss` — иначе
они не попали бы во вторую ветку.

### 3.3. `SupervisedLearner` согласует формы вручную

`avatar/pipeline/tabular/supervised.py:146` — `logits.squeeze(1)` и
`targets.float()`. Это упрощённый пересказ `ClassificationLoss.align`, вшитый
в пайплайн и жёстко предполагающий бинарную задачу. Слой, который не должен
знать про критерий, знает про него ровно настолько, чтобы ошибиться на
регрессии.

### 3.4. `DirectUpliftLoss` не работал ни в одном из трёх смыслов

```
>>> DirectUpliftLoss(1.0)
TypeError: DirectUpliftLoss.__init__() takes 1 positional argument but 2 were given
```

`super().__init__(self)` — раз; `torch.tensor()` без аргументов — два;
сигнатура `forward(treatment_logits, control_logits, is_treat, targets)` не
совпадает ни с одной из веток `SLearner` — три. Класс экспортировался из
`avatar.losses` и стоял в двух таблицах документации. Удалён.

### 3.5. `ContrastiveLoss(separate_heads=True)` терял половину слагаемого

```python
ce_loss = self.c_loss(logits[is_treat == 0], targets[is_treat == 0])
+self.t_loss(logits[is_treat == 1], targets[is_treat == 1])
```

Вторая строка — отдельное выражение с унарным плюсом: значение считалось и
выбрасывалось. Измеренный эффект на синтетическом батче: возвращалось
`0.738733` вместо `1.298693`, а **норма градиента на строках воздействия была
ровно нулевой**. `separate_heads` — это как раз режим, в котором у воздействия
своя голова, так что эта голова не получала от кросс-энтропии ничего.
Исправлено, покрыто двумя тестами (`tests/pipeline/test_losses.py`).

### 3.6. Мелочи, исправленные заодно

* `print()` внутри `forward` у `GradNormLossBalancer`, `KLDxContrastiveLoss` и
  `KLDxContrastiveGridLoss` — печать тензора на каждом шаге обучения.
* `GradNormLossBalancer` существовал в двух копиях (`kld_loss.py` и
  `multi_task.py`); вторая ушла вместе с файлом.

### 3.7. Композиция объявлена, но не используется

`TabularWithAggregatedStates` — это ровно «представляющий стек» из §1. Из трёх
пайплайнов его использует один. `SLearner` и `SupervisedLearner` собирают
embedding, encoder и агрегацию у себя внутри, и как следствие
`SupervisedLearner` на **82%** (109 из 133 строк кода) построчно совпадает с
`SLearner`: это `SLearner` без эмбеддинга воздействия.

---

## 4. Что предлагается дальше

Здесь **не сделано** — это отдельная работа, меняющая публичную поверхность
конфигов.

* **П1.** Один протокол. Всё в `avatar/losses` наследует `Loss` и возвращает
  `LossOutput`; пайплайны зовут функцию потерь только по именованным
  аргументам — тем, что пайплайн в принципе может предложить (`logits`,
  `targets`, `model`, а для uplift ещё `is_treat` и `representation`).
  Функция потерь берёт нужное и глотает остальное через `**kwargs`. Проверка
  по `isinstance` исчезает сама.
* **П2.** Адаптер `TorchCriterion(Loss)`, оборачивающий любой критерий из
  `torch.nn`, — чтобы `nn.BCEWithLogitsLoss` оставался доступен из конфига без
  спецслучая.
* **П3.** Один ключ конфига: `loss:` везде, `loss_fn:` — с предупреждением
  об устаревании.
* **П4.** `SupervisedLearner` — либо собрать его из
  `TabularWithAggregatedStates`, либо удалить: рабочего конфига у него нет ни
  одного, а его постановку закрывает `TabularClassification` (см.
  `docs/reference/pipeline.md`).
* **П5.** Решить судьбу осиротевшего из §0 — `MultiLossMetric`,
  `CollectEmbeddings`, `num_items`.

П1–П3 стоит делать одним заходом: по отдельности каждый оставляет два
протокола в живых.
