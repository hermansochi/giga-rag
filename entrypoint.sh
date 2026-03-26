#!/bin/bash
# entrypoint.sh

set -e  # Выход при ошибке

echo "🚀 Запуск инициализации приложения..."
# === Очищаем старый кэш Streamlit ===
echo "🧹 Очистка кэша Streamlit..."
rm -rf ~/.streamlit

# === Запуск Streamlit ===
echo "🟢 Все модели загружены. Запуск Streamlit..."
exec streamlit run _app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --browser.serverAddress=app.localhost \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false