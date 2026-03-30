"""
pages/3_Мониторинг.py

Мониторинг RAG-системы
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
from collections import defaultdict

from src.database import get_db_connection
from src.config import settings

st.set_page_config(
    page_title="Мониторинг",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Мониторинг RAG-системы")

conn = get_db_connection()

# ====================== 1. БАЛАНС И ТОКЕНЫ ======================
st.subheader("💰 Баланс и использование токенов GigaChat")

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                timestamp,
                total_tokens,
                prompt_tokens,
                completion_tokens,
                balance_entries
            FROM token_usage_log 
            ORDER BY timestamp DESC 
            LIMIT 150
        """)
        log_rows = cur.fetchall()

    st.write(f"**Всего записей в логах:** {len(log_rows)}")

    token_data = []
    balance_by_model = defaultdict(list)

    for row in log_rows:
        ts = str(row['timestamp'])[:19] if row['timestamp'] else "Неизвестно"
        total = row['total_tokens'] if row['total_tokens'] is not None else 0

        token_data.append({
            'timestamp': ts,
            'total_tokens': total,
            'prompt_tokens': row['prompt_tokens'] or 0,
            'completion_tokens': row['completion_tokens'] or 0
        })

        if row['balance_entries']:
            try:
                data = row['balance_entries']
                if isinstance(data, str):
                    data = json.loads(data)

                if isinstance(data, dict) and 'balance' in data:
                    for item in data['balance']:
                        model_name = item.get('usage', 'unknown')
                        value = float(item.get('value', 0))
                        balance_by_model[model_name].append({
                            'timestamp': ts,
                            'value': value
                        })
            except:
                pass

    col1, col2, col3 = st.columns(3)
    total_all = sum(item['total_tokens'] for item in token_data)
    avg_tokens = int(sum(item['total_tokens'] for item in token_data) / len(token_data)) if token_data else 0

    col1.metric("Всего токенов использовано", f"{total_all:,}")
    col2.metric("Среднее токенов на запрос", f"{avg_tokens:,}")
    col3.metric("Всего запросов", len(log_rows))

    if token_data:
        df_tokens = pd.DataFrame(token_data)
        
        # Преобразуем timestamp и группируем по дням (чтобы график не был слишком плотным)
        df_tokens['timestamp'] = pd.to_datetime(df_tokens['timestamp'])
        df_tokens['date'] = df_tokens['timestamp'].dt.date   # группируем только по дате
        
        # Агрегируем по дням
        daily_tokens = df_tokens.groupby('date').sum(numeric_only=True).reset_index()
        daily_tokens = daily_tokens.sort_values('date')

        st.write(f"**Количество дней с данными:** {len(daily_tokens)}")

        # === STACKED BAR — надёжный вариант ===
        fig_tokens = px.bar(
            daily_tokens,
            x='date',
            y=['prompt_tokens', 'completion_tokens'],
            title="Расход токенов по дням (stacked)",
            template="plotly_white",
            barmode='stack',
            text_auto=True,
            color_discrete_sequence=['#1f77b4', '#ff7f0e'],  # синий + оранжевый
            labels={
                "date": "Дата",
                "value": "Количество токенов",
                "variable": "Тип токенов"
            }
        )
        
        fig_tokens.update_layout(
            xaxis_title="Дата",
            yaxis_title="Количество токенов",
            legend_title="Тип токенов",
            height=520,
            bargap=0.05
        )
        
        fig_tokens.update_xaxes(tickangle=45)

        st.plotly_chart(fig_tokens, use_container_width=True)

        st.caption("Синий = prompt_tokens, Оранжевый = completion_tokens")
    else:
        st.warning("Нет данных для построения графика токенов.")
    
    st.subheader("📉 Изменение баланса по моделям")

    if balance_by_model:
        for model_name, history in balance_by_model.items():
            if not history:
                continue
            df_model = pd.DataFrame(history)
            fig = px.line(
                df_model,
                x='timestamp',
                y='value',
                title=f"Баланс модели: **{model_name}**",
                template="plotly_white",
                markers=True
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Пока нет данных по балансу моделей.")

except Exception as e:
    st.error(f"Ошибка загрузки логов токенов: {e}")

st.divider()

# ====================== 5. ЛОГ ЧАТА ======================
st.subheader("💬 История чата")

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                user_message,
                LEFT(assistant_response, 100) || '...' as preview,
                model_name,
                use_rag,
                total_tokens,
                response_time_sec,
                timestamp
            FROM chat_logs 
            ORDER BY timestamp DESC 
            LIMIT 50
        """)
        rows = cur.fetchall()

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Пока нет записей в логах чата.")

except Exception as e:
    if conn:
        conn.rollback()
    st.error(f"Ошибка загрузки логов чата: {e}")

st.subheader("🔍 Детали RAG-запросов")

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                user_message,
                LEFT(assistant_response, 100) || '...' as answer_preview,
                use_rag,
                retrieved_chunks,
                timestamp
            FROM chat_logs 
            WHERE use_rag = true
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        rows = cur.fetchall()

    if rows:
        for row in rows:
            with st.expander(f"💬 {row['user_message'][:80]}... ({row['timestamp']})"):
                st.write("**Найденные источники:**")
                for i, chunk in enumerate(row['retrieved_chunks']):
                    with st.container():
                        st.caption(f"📄 `{chunk['filename']}` | чанк {chunk['chunk_index']} | dist {chunk['distance']:.4f}")
                        st.text(chunk['text'])
    else:
        st.info("Пока нет RAG-запросов.")
except Exception as e:
    if conn:
        conn.rollback()
    st.error(f"Ошибка загрузки RAG-логов: {e}")

st.divider()

st.caption(f"Последнее обновление: {datetime.now().strftime('%H:%M:%S')}")