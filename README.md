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

- Создать VENV
- Отредактиовать config.py под свои модели
- Предзагрузить модели: python3 preload_model.py
- Собрать и запустить: docker compose up -d --build
- Первая сборка может занять длительное время

После запуска приложение будет доступно по адресу:
→ [http://app.localhost](http://app.localhost)
→ [http://localhost:9001/login](http://localhost:9001/login) - UI MinIO
→ [http://s3.localhost](ttp://s3.localhost) - MinIO API
→ [http://pgadmin.localhost](http://pgadmin.localhost) - UI PGAdmin
→ [http://traefik.localhost](http://traefik.localhost) - Traefik dashboard

📁 Баги
- Неправильная логика дедупдикации. Нужно просто проверять существование хеша по тексту чанка и модели эмбедера, если записей нет, только в этом случае генерировать новый эмбединг. 
- Циклический импорт в src/rag/reranker.py
- Разрешен множественный выбор файлов для загрузки, но не реализован.


🔮 Планы развития
Продвинутый чанкинг с извлечением таблиц и структуры (Docling / LlamaParse)
Query rewriting и multi-query retrieval


Ссылка на демонстрацию работы [https://disk.yandex.ru/i/k8SbH6JNzvAUeQ](https://disk.yandex.ru/i/k8SbH6JNzvAUeQ)