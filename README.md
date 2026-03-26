# 🤖 RAG-система на GigaChat

Современная Retrieval-Augmented Generation система с использованием GigaChat (Сбер), PostgreSQL + pgvector, MinIO и Traefik.

Проект позволяет загружать PDF-документы, векторизовать их и вести интеллектуальный чат с поддержкой реранкинга и детального мониторинга.

## ✨ Основные возможности

- 💬 Интеллектуальный чат с документами (RAG)
- 📤 Загрузка PDF с сохранением оригиналов в MinIO
- 🔄 Два режима реранкинга: LLM (GigaChat) и Cross-Encoder
- 📊 Подробный мониторинг использования токенов и документов
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
2. Запуск проекта
Bashmake init          # Полная инициализация (рекомендуется при первом запуске)
После запуска основное приложение будет доступно по адресу:
→ http://app.localhost

Основные команды Make

Команда,Описание
make init,Полная инициализация проекта
make up,Запустить сервисы
make down,Остановить сервисы
make refresh,Очистить кэш Streamlit + перезапустить
make rebuild,Пересобрать Docker-образы
make clear-cache,Очистить только кэш Streamlit
make shell,Зайти в контейнер
make logs,Показать логи


📁 Структура проекта
text.
├── app.py
├── pages/
│   ├── 1_Чат.py
│   ├── 2_Загрузить_файл.py
│   ├── 3_Мониторинг.py
├── src/
│   ├── config.py
│   ├── database.py
│   ├── gigachat.py
│   └── rag/
├── .env
├── docker-compose.yaml
├── Makefile
└── README.md

🔮 Планы развития
Продвинутый чанкинг с извлечением таблиц и структуры (Docling / LlamaParse)
Query rewriting и multi-query retrieval