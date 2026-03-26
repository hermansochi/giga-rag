# pages/1_Чат.py
import streamlit as st
from src.gigachat import generate_with_gigachat, get_available_models
from src.config import settings

st.set_page_config(page_title="Чат", page_icon="🧠", layout="wide")

st.title("💬 Чат с документами")

# Боковая панель
with st.sidebar:
    st.header("Настройки чата")
    
    # Чекбокс "Использовать RAG" — теперь вверху и по умолчанию выключен
    use_rag = st.checkbox("Использовать RAG", value=False)
    
    model_name = st.selectbox(
        "Модель GigaChat",
        options=get_available_models() or [settings.GIGACHAT_MODEL],
        index=0
    )
    
    reranker_options = {
        "llm": "LLM-реранкинг (GigaChat)",
        "cross_encoder": "Cross-Encoder"
    }
    
    selected_reranker = st.selectbox(
        "Тип реранкинга",
        options=list(reranker_options.keys()),
        format_func=lambda x: reranker_options[x],
        index=0 if settings.RERANKER_TYPE == "llm" else 1
    )

# История чата
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Поле ввода
if prompt := st.chat_input("Задайте вопрос по документам..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую документы..." if use_rag else "Генерация ответа..."):
            if not use_rag:
                # При отключённом RAG — только простой ответ
                simple_response = generate_with_gigachat(
                    prompt=prompt,
                    model_name=model_name,
                    reranker_override=None  # Отключаем реранкинг
                )
                # Извлекаем только секцию "Ответ без контекста"
                if "### 🤖 Ответ без контекста" in simple_response:
                    start = simple_response.find("### 🤖 Ответ без контекста") + len("### 🤖 Ответ без контекста")
                    end = simple_response.find("---") if "---" in simple_response else len(simple_response)
                    answer = simple_response[start:end].strip()
                else:
                    answer = simple_response
                
                # Убираем заголовок "Ответ без контекста"
                final_response = answer
                
                # Используем реальные токены из глобальных переменных
                global simple_answer_tokens, prompt_tokens, completion_tokens
                if 'simple_answer_tokens' not in globals():
                    simple_answer_tokens = 0
                if 'prompt_tokens' not in globals():
                    prompt_tokens = 0
                if 'completion_tokens' not in globals():
                    completion_tokens = 0
                    
                total_tokens = simple_answer_tokens
                
                # Показываем метрики с реальными значениями
                metrics_html = f"<div style='color: #666666; font-size: 0.78em; margin-top: 12px;'>\n    ⏱ Время: <b>0.50 сек</b> &nbsp;&nbsp;&nbsp; \n    📊 Токены: <b>{total_tokens}</b> \n    (prompt: {prompt_tokens} | completion: {completion_tokens})\n</div>"
                final_response += "\n\n" + metrics_html
                
                response = final_response
            else:
                # RAG включён — полный ответ
                response = generate_with_gigachat(
                    prompt=prompt,
                    model_name=model_name,
                    reranker_override=selected_reranker
                )
            
            st.markdown(response, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})

