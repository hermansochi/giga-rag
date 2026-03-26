import streamlit as st
import os

st.title("ℹ️ О проекте")

# --- Определяем путь к README.md ---
# Вариант 1: в той же директории, что и скрипт
CURRENT_DIR = os.path.dirname(__file__)
README_PATH = os.path.join(CURRENT_DIR, "..", "README.md")

# Вариант 2: абсолютный путь (на случай, если сборка другая)
ALTERNATIVE_PATH = "/app/README.md"

# Пытаемся найти файл
content = None

for path in [README_PATH, ALTERNATIVE_PATH]:
    path = os.path.normpath(path)  # нормализуем путь
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            st.success(f"📄 Найден `README.md` по пути: `{path}`")
            break
        except Exception as e:
            st.error(f"❌ Ошибка чтения `{path}`: {e}")
            content = None

# --- Отображаем результат ---
if content:
    st.markdown("---")  # разделитель
    st.markdown(content, unsafe_allow_html=True)  # поддержка HTML, если есть
else:
    st.warning("⚠️ Файл `README.md` не найден.")
    st.code("""
    Убедитесь, что:
    1. Файл README.md лежит в корне проекта
    2. Он скопирован в Docker-образ
    3. Пересоберите образ: docker-compose build
    """)