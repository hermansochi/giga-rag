"""
pages/5_О_проекте.py

Страница "О проекте".
Отображает содержимое README.md из корня проекта.
Пытается найти файл по нескольким возможным путям (для локальной разработки и Docker).
"""

import streamlit as st
import os


st.title("ℹ️ О проекте")


# Определяем возможные пути к README.md
CURRENT_DIR = os.path.dirname(__file__)
README_PATH = os.path.join(CURRENT_DIR, "..", "README.md")
ALTERNATIVE_PATH = "/app/README.md"

content = None

for path in [README_PATH, ALTERNATIVE_PATH]:
    path = os.path.normpath(path)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            st.success(f"📄 Найден `README.md` по пути: `{path}`")
            break
        except Exception as e:
            st.error(f"❌ Ошибка чтения `{path}`: {e}")
            content = None

# Отображаем результат
if content:
    st.markdown("---")
    st.markdown(content, unsafe_allow_html=True)
else:
    st.warning("⚠️ Файл `README.md` не найден.")
    st.code("""
    Убедитесь, что:
    1. Файл README.md лежит в корне проекта
    2. Он скопирован в Docker-образ
    3. Пересоберите образ: docker-compose build
    """)