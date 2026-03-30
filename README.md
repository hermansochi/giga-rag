# 🤖 RAG-система на GigaChat

Proof-of-concept (А может даже MVP) Retrieval-Augmented Generation системы с использованием GigaChat (Сбер), PostgreSQL + pgvector, MinIO и Traefik.

Из особенностей - возможность делать реранк разными алгоритмами. Сейчас реализован LLM rerank и cross-encoder на предварительно загруженной в контейнер модели DiTy/cross-encoder-russian-msmarco (на большее памяти моего ноута не хватило).

Проект позволяет загружать несколько типов документов, векторизовать их и вести интеллектуальный чат с поддержкой реранкинга и зачатками мониторинга. В случае необходимости можно оперативно удалить документ. Так же можно менять большинство системных промптов и температуры в разных сценариях использования.

Таким образом данную систему можно потенциально использовать для подбора параметров production RAG.

## ✨ Основные возможности

- 💬 Интеллектуальный чат с документами (RAG)
- 📤 Загрузка PDF, TXT, TEXT, MD, CSV, JSON, JSONL с сохранением оригиналов в MinIO
- 🔄 Два режима реранкинга: LLM (GigaChat) и Cross-Encoder
- 📊 Мониторинг использования токенов и документов
- 🗃️ Расширенные метаданные в базе (JSONB)
- ⚙️ Гибкие настройки через `.env` и интерфейс

## 🛠️ Технологии

| Компонент          | Технология                          |
|--------------------|-------------------------------------|
| UI / Frontend      | Streamlit                           |
| LLM                | GigaChat-2-Max + GigaEmbeddings     |
| Vector Database    | PostgreSQL + pgvector               |
| Object Storage     | MinIO (S3-совместимый)              |
| Reverse Proxy      | Traefik v2                          |
| Оркестрация        | Docker Compose                      |

## 🚀 Быстрый старт

### 1. Настройка
```bash
cp .env.example .env
Отредактируйте файл .env — обязательно укажите ваш GIGACHAT_API_KEY.

После запуска основное приложение будет доступно по адресу:
→ http://app.localhost
→ http://localhost:9001/login - UI MinIO
→ http://s3.localhost - MinIO API
→ http://pgadmin.localhost - UI PGAdmin
→ http://traefik.localhost/ - Traefik dashboard


📁 Баги
- Циклический импорт в src/rag/reranker.py
- Разрешен множественный выбор файлов для загрузки, но не реализован.


🔮 Планы развития
Продвинутый чанкинг с извлечением таблиц и структуры (Docling / LlamaParse)
Query rewriting и multi-query retrieval

Оценка проекта от GigaCode в агентном режиме:
