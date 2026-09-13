# Своя функция потерь

Минимальный рабочий пример точки расширения «функция потерь»: `loss.py`
содержит `FocalLoss` — focal loss для несбалансированной бинарной
классификации, который занижает вклад примеров, уже уверенно предсказанных
моделью.

Контракт — в [../../docs/guides/losses.md](../../docs/guides/losses.md).

## Подключение

```yaml
model:
  _target_: avatar.pipeline.tabular.TabularClassification
  num_classes: 2
  tabular_model: ...
  loss:
    _target_: examples.custom_loss.loss.FocalLoss
    gamma: 2.0
```

Пайплайн создаёт функцию потерь сам; ключ `loss:` её подменяет.

## Что является контрактом, а что нет

Контракт ровно один: **вернуть `LossOutput`**.

```python
LossOutput(
    loss=focal,                                    # скаляр для backward()
    components={"cross_entropy": ...},             # слагаемые для логов
)
```

Список аргументов контрактом **не** является. Его задаёт пайплайн, потому что
только он знает, что значат его таргеты. `TabularClassification` вызывает
функцию потерь как `self.loss(logits, targets, model=self.encoder)` — значит,
замена обязана принимать такую сигнатуру. Поэтому `FocalLoss` принимает
`model=None` и игнорирует его.

У `SLearner` та же подмена устроена иначе: ключ называется `loss_fn:`, ждёт
голый `nn.Module` и вызывается с `(logits, is_treat, dist, targets)`. Подробнее
— в `docs/guides/losses.md`, раздел «Два протокола».

## Про `components`

Слагаемые нужны не для оптимизации, а для наблюдаемости: их подхватят
`avatar.metrics.MultiLossMetric` и `avatar.metrics.UniversalLossesMetric`. В
примере туда кладётся необработанная кросс-энтропия — чтобы по логам было видно,
насколько именно focal-взвешивание изменило значение.

Кладите в `components` только `detach()`-нутые тензоры: они переживают батч, и
удержанный граф — это утечка памяти.

## Проверка

```bash
python -m pytest tests/examples
```

Среди прочего проверяется, что при `gamma > 0` итоговое значение меньше
кросс-энтропии — то есть взвешивание действительно работает, а не просто
считается.
