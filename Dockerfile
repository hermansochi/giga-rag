FROM python:3.11

# Устанавливаем uv (очень быстро)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Копируем только requirements.txt первым (для лучшего кэширования слоёв)
COPY requirements.txt .

# Установка зависимостей через uv с кэшированием
RUN --mount=type=cache,target=/root/.cache/uv \
    echo "=== Начало установки зависимостей через uv ===" && \
    uv pip install --system -r requirements.txt && \
    echo "=== Установка зависимостей успешно завершена ==="

# Копируем весь остальной код проекта
COPY . .

# Делаем entrypoint исполняемым
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8501

CMD ["/app/entrypoint.sh"]