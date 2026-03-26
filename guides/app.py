import os
import torch
import argparse
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import streamlit as st

# === Парсинг аргументов ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Hugging Face model name")
parser.add_argument("--use_gigachat", action="store_true", help="Use GigaChat API instead of local model")
parser.add_argument("--gigachat_model", type=str, default="GigaChat-2-Max", help="GigaChat model (e.g., GigaChat-2)")
args = parser.parse_args()

# === Настройка модели ===
model_name = args.model or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
# Загружаем локальную модель и токенизатор (если нужен локальный режим)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Инициализация GigaChat (если нужно) ===
gigachat_api_key = os.getenv("GIGACHAT_API_KEY", "")
use_gigachat = args.use_gigachat or bool(gigachat_api_key)


# === Функции генерации ===
def generate_with_local(model, tokenizer, user_input):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def generate_with_gigachat(prompt, model_name="GigaChat-2-Max"):
    """
    Генерирует ответ через GigaChat API.
    """

    # Формируем сообщения
    messages = [
        Messages(
            role=MessagesRole.SYSTEM,
            content="Ты очень полезный помощник. Ты всегда отвечаешь в стихах."
        ),
        Messages(
            role=MessagesRole.USER,
            content=prompt
        )
    ]

    # Отправляем запрос к GigaChat
    with GigaChat(credentials=gigachat_api_key, scope='GIGACHAT_API_B2B', verify_ssl_certs=False, model=model_name) as giga:
        response = giga.chat(Chat(messages=messages))

    # Извлекаем текст ответа
    return response.choices[0].message.content.strip()

# === Gradio-интерфейс ===
# def chat_interface(user_input):
#     if use_gigachat:
#         return generate_with_gigachat(user_input)
#     else:
#         return generate_with_local(model, tokenizer, user_input)

# iface = gr.Interface(
#     fn=chat_interface,
#     inputs=gr.Textbox(
#         lines=4,
#         placeholder="Введите ваш запрос здесь...",
#         label="Входное сообщение",
#         show_copy_button=True
#     ),
#     outputs=gr.Textbox(
#         lines=12,
#         label="Ответ модели",
#         show_copy_button=True,
#         show_label=True,
#         max_lines=20
#     ),
#     title="LLM Chat Application",
#     description="🚀 Введите ваш запрос в поле ниже и получите ответ от языковой модели",
#     theme="soft",
#     allow_flagging="never"
# )
# iface.launch()  # Для запуска Gradio

# === Streamlit-интерфейс ===
st.set_page_config(page_title="LLM Chat App", layout="centered")
st.title("LLM Chat App")

user_input = st.text_area("Введите запрос:", height=100)
if st.button("Сгенерировать"):
    if not user_input:
        st.warning("Пожалуйста, введите запрос.")
    else:
        if use_gigachat:
            answer = generate_with_gigachat(user_input)
        else:
            answer = generate_with_local(model, tokenizer, user_input)
        st.subheader("Ответ модели:")
        st.write(answer)
