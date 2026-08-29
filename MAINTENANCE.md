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
