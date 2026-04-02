"""
pages/3_Мониторинг.py

Улучшенная страница мониторинга RAG-системы с вкладками.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
from collections import defaultdict

from src.database import get_db_connection

st.set_page_config(page_title="Мониторинг", page_icon="📊", layout="wide")

st.title("📊 Мониторинг RAG-системы")


# ====================== Вспомогательные функции ======================
@st.cache_data(ttl=30)
def get_token_logs(limit: int = 200) -> pd.DataFrame:
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT timestamp, total_tokens, prompt_tokens, completion_tokens, balance_entries
            FROM token_usage_log 
            ORDER BY timestamp DESC 
            LIMIT %s
        """,
            (limit,),
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=60)
def get_table_stats() -> pd.DataFrame:
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                t.table_name,
                pg_size_pretty(pg_total_relation_size(t.table_schema || '.' || t.table_name)) AS total_size,
                pg_total_relation_size(t.table_schema || '.' || t.table_name) AS total_size_bytes
            FROM information_schema.tables t
            WHERE t.table_schema = 'public'
            ORDER BY total_size_bytes DESC;
        """)
        tables = cur.fetchall()

    stats = []
    for row in tables:
        table_name = row["table_name"]
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()["count"]
        stats.append(
            {"Таблица": table_name, "Записей": int(count), "Размер": row["total_size"]}
        )
    return pd.DataFrame(stats)


# ====================== Основной интерфейс ======================
tab1, tab2, tab3 = st.tabs(["📈 Токены и баланс", "📋 Таблицы БД", "💬 История чата"])

with tab1:
    st.subheader("💰 Использование токенов")

    df_tokens = get_token_logs()

    if not df_tokens.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего токенов", f"{df_tokens['total_tokens'].sum():,}")
        col2.metric(
            "Среднее на запрос",
            f"{int(df_tokens['total_tokens'].mean()):,}" if len(df_tokens) > 0 else 0,
        )
        col3.metric("Всего запросов", len(df_tokens))

        # Stacked bar по дням
        df_tokens["date"] = pd.to_datetime(df_tokens["timestamp"]).dt.date
        daily = df_tokens.groupby("date").sum(numeric_only=True).reset_index()

        fig_tokens = px.bar(
            daily,
            x="date",
            y=["prompt_tokens", "completion_tokens"],
            title="Расход токенов по дням (stacked)",
            template="plotly_white",
            barmode="stack",
            text_auto=True,
        )
        fig_tokens.update_layout(height=520)
        st.plotly_chart(fig_tokens, use_container_width=True)

        # ====================== ГРАФИК БАЛАНСА ======================
        st.subheader("📉 Изменение баланса по моделям")

        balance_by_model = defaultdict(list)

        for _, row in df_tokens.iterrows():
            if row["balance_entries"]:
                try:
                    data = row["balance_entries"]
                    if isinstance(data, str):
                        data = json.loads(data)

                    if isinstance(data, dict) and "balance" in data:
                        for item in data["balance"]:
                            model_name = item.get("usage", "unknown")
                            value = float(item.get("value", 0))
                            balance_by_model[model_name].append(
                                {"timestamp": row["timestamp"], "value": value}
                            )
                except:
                    pass

        if balance_by_model:
            for model_name, history in balance_by_model.items():
                if not history:
                    continue
                df_model = pd.DataFrame(history)
                fig = px.line(
                    df_model,
                    x="timestamp",
                    y="value",
                    title=f"Баланс модели: **{model_name}**",
                    template="plotly_white",
                    markers=True,
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Пока нет данных по балансу моделей.")
    else:
        st.info("Пока нет данных по токенам.")

with tab2:
    st.subheader("📋 Таблицы базы данных")
    df_tables = get_table_stats()

    if not df_tables.empty:
        col1, col2 = st.columns(2)
        col1.metric("Всего таблиц", len(df_tables))
        col2.metric("Всего записей", f"{df_tables['Записей'].sum():,}")

        st.dataframe(
            df_tables.sort_values("Записей", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            df_tables.sort_values("Записей", ascending=False),
            x="Таблица",
            y="Записей",
            title="Количество записей по таблицам",
            text="Записей",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Не удалось получить статистику таблиц.")

with tab3:
    st.subheader("💬 История чата")
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                user_message,
                LEFT(assistant_response, 120) || '...' as preview,
                model_name,
                use_rag,
                total_tokens,
                response_time_sec,
                timestamp
            FROM chat_logs 
            ORDER BY timestamp DESC 
            LIMIT 30
        """)
        rows = cur.fetchall()

    if rows:
        df_chat = pd.DataFrame(rows)
        st.dataframe(df_chat, use_container_width=True, hide_index=True)
    else:
        st.info("Пока нет записей в истории чата.")

st.divider()
st.caption(f"Последнее обновление: {datetime.now().strftime('%H:%M:%S')}")
