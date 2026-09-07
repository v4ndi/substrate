# Чистка репозитория

Задание от 2026-08-29. **Выполнено.** Удалены неиспользуемые модули/реализации.

---

## 1. `avatar/nn/augmentations` — удалено

Каталог удалён целиком (`__init__.py`, `mixup.py` с `MixupEmbeddingProcessor`).
Нигде в `avatar/`, `tests/`, `examples/` не импортировался, кроме реэкспорта
в `avatar/nn/__init__.py` — оттуда убран.

## 2. `avatar/nn/embedding/embeddings.py` — удалён `PLEEmbedding`

- Класс `PLEEmbedding` (обёртка над `rtdl_num_embeddings.PiecewiseLinearEmbeddings`)
  удалён вместе с неиспользуемым после этого импортом `yaml`.
- Реэкспорт `PLEEmbedding` убран из `avatar/nn/embedding/__init__.py`.
- Зависимость `rtdl_num_embeddings==0.0.12` убрана из `pyproject.toml`
  (единственное её использование было в `PLEEmbedding`).

## 3. `avatar/nn/embedding/universal_embedding.py` — удалено

Файл удалён (`UniversalEmbedding`, `UniversalEventEmbedding`). Импорт
`UniversalEventEmbedding` убран из `avatar/nn/embedding/__init__.py`.
Нигде больше не использовался.

---

**Проверки:** `import avatar.nn` OK; `ruff check` по изменённым файлам чист;
полный прогон тестов — `144 passed, 66 deselected` (без регрессий относительно
предыдущего состояния).

---

# Рефакторинг `avatar/nn/embedding`

Задание от 2026-08-29. **Выполнено.** Плоский модуль разнесён на подпакеты.

Пакет остаётся импортируемым как `avatar.nn.embedding` — `__init__.py` реэкспортит
весь публичный API, поэтому ~60 Hydra-конфигов (`_target_: avatar.nn.embedding.*`)
и внешние импорты не тронуты.

## Новая раскладка

```
avatar/nn/embedding/
├── __init__.py            # плоский реэкспорт публичного API
├── base/                  # общее для sequential и tabular
│   ├── embedding.py       # BaseEmbedding
│   └── primitives.py      # _check_input_shape, LinearEmbeddings, HashEmbedding
├── sequential/            # event-sequence эмбеддинги
│   ├── base.py            # BaseEventSequenceEmbedding, BaseTemporalEmbedding
│   ├── event.py           # EventSequenceEmbedding
│   └── position.py        # TemporalPositionEncoding, Time2VecEmbedding
└── tabular/               # tabular эмбеддинги
    ├── base.py            # BaseTabularEmbedding
    ├── numeric.py         # NumericFeatureEmbedding (бывш. NumEmbedding)
    ├── embedding.py       # TabularEmbedding
    └── hidden_state_agg.py  # BaseHiddenStateAggregator, LayerNorm{Concatenate,Sum}
```

Было: `base_embedding.py`, `embeddings.py`, `hash_embedding.py`,
`hidden_state_agg.py`, `position_embeddings.py` (все удалены).

## Прочие изменения

- `NumEmbedding` → `NumericFeatureEmbedding` (класс был непубличным, использовался
  только в `TabularEmbedding`); теперь экспортируется из `avatar.nn.embedding`.
- Доменные базовые классы уехали в свои подпакеты (`BaseEventSequenceEmbedding`
  в `sequential/`, `BaseTabularEmbedding` в `tabular/`); в `base/` — только
  действительно общее.
- 4 внутренних потребителя (`feature_encoder/{base_feature_encoder,attention_encoder}.py`,
  `pipeline/{tabular/supervised,uplift/s_learner}.py`) переведены с глубоких
  импортов на top-level `from avatar.nn.embedding import ...`.

**Проверки:** импорт `avatar.nn.embedding` + всех 4 потребителей OK; `ruff check`
и `ruff format --check` чисты; тесты `144 passed, 66 deselected` (без регрессий).
