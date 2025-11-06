import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# АНАЛИЗАТОР РЕАЛЬНЫХ ДАННЫХ
# ============================================================================

class SalesDataAnalyzer:
    """Класс для анализа реальных данных продаж"""
    
    def __init__(self, df):
        self.df = self._prepare_data(df)
        self.salons_stats = self._calculate_salon_stats()
        self.clusters = self._create_clusters()
        
    def _prepare_data(self, df):
        """Подготовка и очистка данных"""
        # Переименование колонок для единообразия
        column_mapping = {
            'Magazin': 'salon',
            'Datasales': 'date',
            'Art': 'article',
            'Describe': 'description',
            'Model': 'model',
            'Segment': 'segment',
            'Purchaiseprice': 'cost_price',
            'Price': 'price',
            'Qty': 'quantity',
            'Sum': 'revenue'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Конвертация даты
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Расчет прибыли и маржи
        if 'cost_price' in df.columns and 'price' in df.columns:
            df['profit'] = (df['price'] - df['cost_price']) * df['quantity']
            df['margin'] = ((df['price'] - df['cost_price']) / df['price'] * 100).clip(0, 100)
        
        return df
    
    def _calculate_salon_stats(self):
        """Расчет статистики по салонам"""
        stats = self.df.groupby('salon').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'price': 'mean',
            'margin': 'mean',
            'date': 'count'  # количество транзакций
        }).rename(columns={'date': 'transactions'})
        
        # Средний чек
        stats['avg_check'] = stats['revenue'] / stats['transactions']
        
        # Маржинальность
        stats['margin_pct'] = (stats['profit'] / stats['revenue'] * 100).fillna(0)
        
        return stats.sort_values('revenue', ascending=False)
    
    def _create_clusters(self):
        """Автоматическая кластеризация салонов"""
        # Простая кластеризация по среднему чеку
        stats = self.salons_stats.copy()
        
        # Квантили для разделения
        q33 = stats['avg_check'].quantile(0.33)
        q66 = stats['avg_check'].quantile(0.66)
        
        def assign_cluster(avg_check):
            if avg_check >= q66:
                return 'A'  # Премиум
            elif avg_check >= q33:
                return 'B'  # Средний
            else:
                return 'C'  # Эконом
        
        stats['cluster'] = stats['avg_check'].apply(assign_cluster)
        
        return stats
    
    def get_segment_analysis(self):
        """Анализ по сегментам товаров"""
        if 'segment' not in self.df.columns:
            return None
            
        segment_stats = self.df.groupby('segment').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'price': 'mean'
        }).sort_values('revenue', ascending=False)
        
        return segment_stats
    
    def get_time_series(self):
        """Временной ряд продаж"""
        if 'date' not in self.df.columns:
            return None
            
        ts = self.df.groupby(self.df['date'].dt.to_period('M')).agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        })
        
        return ts

# ============================================================================
# СИМУЛЯТОР НА ОСНОВЕ РЕАЛЬНЫХ ДАННЫХ
# ============================================================================

class RealDataSimulator:
    """Симулятор на основе реальных данных"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.baseline = analyzer.salons_stats
        
    def simulate_price_change(self, price_change_pct, target_cluster, selected_segment=None):
        """
        Симуляция изменения цен
        
        Параметры:
        - price_change_pct: изменение цены в %
        - target_cluster: кластер для изменения ('A', 'B', 'C')
        - selected_segment: конкретный сегмент товаров (опционально)
        """
        results = []
        
        # Эластичности по кластерам
        elasticity = {
            'A': -0.8,   # Премиум: низкая эластичность
            'B': -1.2,   # Средний: средняя эластичность
            'C': -1.5    # Эконом: высокая эластичность
        }
        
        # Эффект перетока
        spillover_to_target = 0.25
        spillover_from_others = 0.03
        
        for salon, baseline_stats in self.baseline.iterrows():
            cluster = self.analyzer.clusters.loc[salon, 'cluster']
            
            if cluster == target_cluster:
                # Салон с изменением цены
                
                # Изменение спроса
                demand_multiplier = 1 + (price_change_pct / 100) * elasticity[cluster]
                
                # Приток клиентов при снижении цен
                if price_change_pct < 0:
                    demand_multiplier += spillover_to_target
                
                # Новая выручка
                new_revenue = baseline_stats['revenue'] * demand_multiplier * (1 + price_change_pct / 100)
                
                # Изменение маржи (при снижении цены маржа падает сильнее)
                margin_drop = abs(price_change_pct) * 1.5 if price_change_pct < 0 else 0
                new_margin_pct = max(baseline_stats['margin_pct'] - margin_drop, 5)
                
                new_profit = new_revenue * (new_margin_pct / 100)
                
            else:
                # Салоны без изменений
                loss_factor = spillover_from_others if cluster == 'B' and price_change_pct < 0 else 0
                
                new_revenue = baseline_stats['revenue'] * (1 - loss_factor)
                new_profit = baseline_stats['profit'] * (1 - loss_factor)
            
            results.append({
                'salon': salon,
                'cluster': cluster,
                'baseline_revenue': baseline_stats['revenue'],
                'new_revenue': new_revenue,
                'baseline_profit': baseline_stats['profit'],
                'new_profit': new_profit,
                'revenue_change_pct': (new_revenue / baseline_stats['revenue'] - 1) * 100,
                'profit_change_pct': (new_profit / baseline_stats['profit'] - 1) * 100
            })
        
        return pd.DataFrame(results)
    
    def get_summary(self, simulation_df):
        """Сводка по симуляции"""
        summary = {
            'total': {
                'baseline_revenue': simulation_df['baseline_revenue'].sum(),
                'new_revenue': simulation_df['new_revenue'].sum(),
                'baseline_profit': simulation_df['baseline_profit'].sum(),
                'new_profit': simulation_df['new_profit'].sum()
            },
            'by_cluster': simulation_df.groupby('cluster').agg({
                'baseline_revenue': 'sum',
                'new_revenue': 'sum',
                'baseline_profit': 'sum',
                'new_profit': 'sum'
            }).to_dict('index')
        }
        
        summary['total']['revenue_change_pct'] = (
            (summary['total']['new_revenue'] / summary['total']['baseline_revenue'] - 1) * 100
        )
        summary['total']['profit_change_pct'] = (
            (summary['total']['new_profit'] / summary['total']['baseline_profit'] - 1) * 100
        )
        
        return summary

# ============================================================================
# STREAMLIT ИНТЕРФЕЙС
# ============================================================================

st.set_page_config(
    page_title="Симуляция 'Что если' - Анализ продаж оптики",
    page_icon="👓",
    layout="wide"
)

st.title("👓 Симуляция 'Что если' для сети салонов оптики")
st.markdown("### На основе ваших реальных данных продаж")

# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

uploaded_file = st.file_uploader("📁 Загрузите Excel файл с историей продаж", type=['xlsx', 'xls'])

if uploaded_file is not None:
    
    # Загрузка данных
    try:
        with st.spinner('Загрузка и анализ данных...'):
            df = pd.read_excel(uploaded_file)
            analyzer = SalesDataAnalyzer(df)
            simulator = RealDataSimulator(analyzer)
        
        st.success(f"✅ Загружено {len(df):,} записей о продажах")
        
        # ====================================================================
        # ВКЛАДКИ
        # ====================================================================
        
        tab1, tab2, tab3 = st.tabs(["📊 Анализ данных", "🎯 Симуляция", "📋 Детали"])
        
        # ====================================================================
        # ВКЛАДКА 1: АНАЛИЗ ДАННЫХ
        # ====================================================================
        
        with tab1:
            st.header("Анализ текущих данных")
            
            # Общая статистика
            col1, col2, col3, col4 = st.columns(4)
            
            total_revenue = analyzer.df['revenue'].sum()
            total_profit = analyzer.df['profit'].sum()
            total_qty = analyzer.df['quantity'].sum()
            avg_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
            
            with col1:
                st.metric("Выручка", f"{total_revenue / 1_000_000:.1f}M₽")
            with col2:
                st.metric("Прибыль", f"{total_profit / 1_000_000:.1f}M₽")
            with col3:
                st.metric("Продано единиц", f"{total_qty:,.0f}")
            with col4:
                st.metric("Средняя маржа", f"{avg_margin:.1f}%")
            
            st.markdown("---")
            
            # Статистика по салонам
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💼 Топ-10 салонов по выручке")
                top_salons = analyzer.salons_stats.head(10)[['revenue', 'profit', 'transactions']]
                top_salons['revenue'] = top_salons['revenue'].apply(lambda x: f"{x/1000:.0f}K₽")
                top_salons['profit'] = top_salons['profit'].apply(lambda x: f"{x/1000:.0f}K₽")
                st.dataframe(top_salons, use_container_width=True)
            
            with col2:
                st.subheader("📊 Распределение по кластерам")
                cluster_dist = analyzer.clusters['cluster'].value_counts()
                
                fig = px.pie(
                    values=cluster_dist.values,
                    names=cluster_dist.index,
                    title="Количество салонов по кластерам",
                    color=cluster_dist.index,
                    color_discrete_map={'A': 'gold', 'B': 'silver', 'C': 'brown'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Анализ по сегментам
            segment_stats = analyzer.get_segment_analysis()
            if segment_stats is not None:
                st.subheader("🏷️ Продажи по сегментам товаров")
                
                fig = px.bar(
                    segment_stats.reset_index(),
                    x='segment',
                    y='revenue',
                    title="Выручка по сегментам",
                    labels={'revenue': 'Выручка (₽)', 'segment': 'Сегмент'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Временной ряд
            ts = analyzer.get_time_series()
            if ts is not None:
                st.subheader("📈 Динамика продаж")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts.index.to_timestamp(),
                    y=ts['revenue'],
                    mode='lines+markers',
                    name='Выручка',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    xaxis_title="Месяц",
                    yaxis_title="Выручка (₽)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ====================================================================
        # ВКЛАДКА 2: СИМУЛЯЦИЯ
        # ====================================================================
        
        with tab2:
            st.header("Сценарий: Изменение ценовой политики")
            
            # Параметры симуляции
            col1, col2 = st.columns(2)
            
            with col1:
                cluster = st.selectbox(
                    "Кластер салонов для изменения цен",
                    options=['A', 'B', 'C'],
                    help="A - Премиум, B - Средний, C - Эконом"
                )
                
                cluster_info = analyzer.clusters[analyzer.clusters['cluster'] == cluster]
                st.info(f"📍 В кластере {cluster}: {len(cluster_info)} салонов")
            
            with col2:
                price_change = st.slider(
                    "Изменение цены (%)",
                    min_value=-30,
                    max_value=30,
                    value=-10,
                    step=5
                )
            
            # Выбор сегмента (если есть)
            segments = analyzer.df['segment'].unique() if 'segment' in analyzer.df.columns else []
            if len(segments) > 0:
                segment = st.selectbox(
                    "Сегмент товаров (опционально)",
                    options=['Все'] + list(segments)
                )
            else:
                segment = 'Все'
            
            # Кнопка запуска
            if st.button("🚀 Запустить симуляцию", type="primary"):
                
                with st.spinner("Расчет..."):
                    selected_segment = None if segment == 'Все' else segment
                    results = simulator.simulate_price_change(price_change, cluster, selected_segment)
                    summary = simulator.get_summary(results)
                
                # Результаты
                st.markdown("---")
                st.subheader("📊 Результаты симуляции")
                
                col1, col2, col3, col4 = st.columns(4)
                
                revenue_change = summary['total']['revenue_change_pct']
                profit_change = summary['total']['profit_change_pct']
                
                with col1:
                    st.metric(
                        "Изменение выручки",
                        f"{revenue_change:+.1f}%",
                        delta=f"{(summary['total']['new_revenue'] - summary['total']['baseline_revenue']) / 1_000_000:.1f}M₽"
                    )
                
                with col2:
                    st.metric(
                        "Изменение прибыли",
                        f"{profit_change:+.1f}%",
                        delta=f"{(summary['total']['new_profit'] - summary['total']['baseline_profit']) / 1_000_000:.1f}M₽"
                    )
                
                with col3:
                    st.metric(
                        "Новая выручка",
                        f"{summary['total']['new_revenue'] / 1_000_000:.1f}M₽"
                    )
                
                with col4:
                    st.metric(
                        "Новая прибыль",
                        f"{summary['total']['new_profit'] / 1_000_000:.1f}M₽"
                    )
                
                # Вердикт
                st.markdown("---")
                if profit_change < 0:
                    st.error(f"⚠️ **ВЕРДИКТ: НЕВЫГОДНО** - Прибыль снижается на {abs(profit_change):.1f}%")
                else:
                    st.success(f"✅ **ВЕРДИКТ: ВЫГОДНО** - Прибыль растет на {profit_change:.1f}%")
                
                # Графики
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Выручка по кластерам")
                    cluster_data = []
                    for cl, data in summary['by_cluster'].items():
                        cluster_data.append({
                            'Кластер': cl,
                            'Базовая': data['baseline_revenue'] / 1_000_000,
                            'Новая': data['new_revenue'] / 1_000_000
                        })
                    
                    df_cluster = pd.DataFrame(cluster_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Базовая', x=df_cluster['Кластер'], y=df_cluster['Базовая']))
                    fig.add_trace(go.Bar(name='Новая', x=df_cluster['Кластер'], y=df_cluster['Новая']))
                    fig.update_layout(barmode='group', yaxis_title='Выручка (млн ₽)')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("💰 Прибыль по кластерам")
                    profit_data = []
                    for cl, data in summary['by_cluster'].items():
                        profit_data.append({
                            'Кластер': cl,
                            'Базовая': data['baseline_profit'] / 1_000_000,
                            'Новая': data['new_profit'] / 1_000_000
                        })
                    
                    df_profit = pd.DataFrame(profit_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Базовая', x=df_profit['Кластер'], y=df_profit['Базовая']))
                    fig.add_trace(go.Bar(name='Новая', x=df_profit['Кластер'], y=df_profit['Новая']))
                    fig.update_layout(barmode='group', yaxis_title='Прибыль (млн ₽)')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Таблица результатов
                st.markdown("---")
                st.subheader("📋 Детализация по салонам")
                
                display_results = results[['salon', 'cluster', 'baseline_revenue', 'new_revenue', 
                                          'revenue_change_pct', 'baseline_profit', 'new_profit', 
                                          'profit_change_pct']].copy()
                
                display_results.columns = ['Салон', 'Кластер', 'Выручка (база)', 'Выручка (новая)',
                                          'Δ Выручка %', 'Прибыль (база)', 'Прибыль (новая)', 'Δ Прибыль %']
                
                for col in ['Выручка (база)', 'Выручка (новая)', 'Прибыль (база)', 'Прибыль (новая)']:
                    display_results[col] = display_results[col].apply(lambda x: f"{x/1000:.0f}K₽")
                
                for col in ['Δ Выручка %', 'Δ Прибыль %']:
                    display_results[col] = display_results[col].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(display_results, use_container_width=True, height=400)
        
        # ====================================================================
        # ВКЛАДКА 3: ДЕТАЛИ
        # ====================================================================
        
        with tab3:
            st.header("Детальная информация о салонах")
            
            # Полная таблица салонов с кластерами
            full_stats = analyzer.clusters.copy()
            full_stats['revenue'] = full_stats['revenue'].apply(lambda x: f"{x/1_000_000:.2f}M₽")
            full_stats['profit'] = full_stats['profit'].apply(lambda x: f"{x/1_000_000:.2f}M₽")
            full_stats['avg_check'] = full_stats['avg_check'].apply(lambda x: f"{x:.0f}₽")
            full_stats['margin_pct'] = full_stats['margin_pct'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(full_stats, use_container_width=True, height=600)
            
            # Информация о модели
            with st.expander("ℹ️ О модели симуляции"):
                st.markdown("""
                **Как работает симуляция:**
                
                1. **Автоматическая кластеризация салонов:**
                   - По среднему чеку делим на 3 кластера (A, B, C)
                   - A = премиум, B = средний, C = эконом
                
                2. **Эластичность спроса:**
                   - A (Премиум): -0.8
                   - B (Средний): -1.2
                   - C (Эконом): -1.5
                
                3. **Эффекты:**
                   - Переток клиентов между кластерами
                   - Изменение маржинальности при изменении цен
                   - Влияние на всю сеть
                
                4. **Расчет:**
                   - Используются ваши реальные данные
                   - Базовые показатели = фактические из истории
                   - Прогноз = модель + эластичности
                """)
    
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке файла: {str(e)}")
        st.info("Убедитесь, что файл содержит колонки: Magazin, Datasales, Price, Qty, Sum и т.д.")

else:
    # Инструкция для пользователя
    st.info("👆 Загрузите Excel файл с историей продаж для начала анализа")
    
    st.markdown("""
    ### 📋 Требования к файлу:
    
    Файл должен содержать следующие колонки:
    - **Magazin** - название салона
    - **Datasales** - дата продажи
    - **Art** - артикул товара
    - **Describe** - описание товара
    - **Model** - модель
    - **Segment** - сегмент товара
    - **Purchaiseprice** - цена закупки
    - **Price** - цена продажи
    - **Qty** - количество
    - **Sum** - сумма продажи
    
    ### 🎯 Что вы получите:
    
    1. **Анализ данных:**
       - Статистика по салонам
       - Распределение по кластерам
       - Анализ сегментов товаров
       - Временные тренды
    
    2. **Симуляция "Что если":**
       - Изменение цен по кластерам
       - Прогноз выручки и прибыли
       - Детализация по салонам
       - Визуализация результатов
    
    3. **Рекомендации:**
       - Выгодно/невыгодно
       - Влияние на сеть
       - Декомпозиция эффектов
    """)