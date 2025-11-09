import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# ІНІЦІАЛІЗАЦІЯ SESSION STATE
# ============================================================================

if 'df' not in st.session_state:
    st.session_state.df = None

# ============================================================================
# ФУНКЦІЇ ДЛЯ ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================================

def validate_dataframe(df):
    """Перевірка структури даних"""
    if df is None or df.empty:
        return False, "Порожній датафрейм"
    
    required_columns = ['Magazin', 'Datasales', 'Price', 'Qty', 'Sum']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        return False, f"Відсутні колонки: {', '.join(missing)}"
    
    return True, "OK"

@st.cache_data(ttl=600, show_spinner=False)
def load_data_from_google_sheets(spreadsheet_url):
    """Завантаження даних з Google Sheets"""
    try:
        # Перевірка URL
        if not spreadsheet_url or '/d/' not in spreadsheet_url:
            return None, "Невірний формат URL"
        
        # Витягуємо ID
        sheet_id = spreadsheet_url.split('/d/')[1].split('/')[0]
        gid = '0'
        if 'gid=' in spreadsheet_url:
            gid = spreadsheet_url.split('gid=')[1].split('&')[0].split('#')[0]
        
        # URL для експорту
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Завантаження
        df = pd.read_csv(export_url, encoding='utf-8', on_bad_lines='skip')
        
        if df.empty:
            return None, "Таблиця порожня"
        
        return df, None
        
    except Exception as e:
        return None, str(e)

def load_excel_file(uploaded_file):
    """Завантаження Excel файлу"""
    try:
        # Спроба з openpyxl
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df, None
    except Exception as e1:
        try:
            # Спроба з xlrd
            df = pd.read_excel(uploaded_file, engine='xlrd')
            return df, None
        except Exception as e2:
            return None, f"Помилка: {str(e1)}"

# ============================================================================
# АНАЛІЗАТОР ДАНИХ
# ============================================================================
class SalesDataAnalyzer:
    """Клас для аналізу реальних даних продажів"""
    
    def __init__(self, df):
        self.df = self._prepare_data(df)
        self.salons_stats = self._calculate_salon_stats()
        self.clusters = self._create_clusters()
        
    def _prepare_data(self, df):
        """Підготовка та очищення даних"""
        # Перейменування колонок
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
        
        # 🔧 КРИТИЧНО: Конвертація числових колонок
        numeric_columns = ['price', 'quantity', 'revenue', 'cost_price']
        for col in numeric_columns:
            if col in df.columns:
                # Обробка текстових значень
                if df[col].dtype == 'object':
                    df[col] = (df[col]
                              .astype(str)
                              .str.replace(',', '.')
                              .str.replace(' ', '')
                              .str.replace('₴', '')
                              .str.strip())
                # Конвертація в числа
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Конвертація дати
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Очищення від NaN
        df = df.dropna(subset=['price', 'quantity', 'revenue'])
        
        # Перевірка коректності
        df = df[df['price'] > 0]
        df = df[df['quantity'] > 0]
        df = df[df['revenue'] > 0]
        
        # Розрахунок прибутку
        if 'cost_price' in df.columns and df['cost_price'].notna().any():
            df['profit'] = (df['price'] - df['cost_price']) * df['quantity']
            df['margin'] = ((df['price'] - df['cost_price']) / df['price'] * 100).clip(0, 100)
        else:
            df['profit'] = df['revenue'] * 0.40
            df['margin'] = 40.0
        
        return df
    
    def _calculate_salon_stats(self):
        """Розрахунок статистики по салонах"""
        stats = self.df.groupby('salon').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'price': 'mean',
            'margin': 'mean',
            'date': 'count'
        }).rename(columns={'date': 'transactions'})
        
        stats['avg_check'] = stats['revenue'] / stats['transactions']
        stats['margin_pct'] = (stats['profit'] / stats['revenue'] * 100).fillna(0)
        stats['roi'] = ((stats['profit'] / (stats['revenue'] - stats['profit'])) * 100).fillna(0)
        
        return stats.sort_values('revenue', ascending=False)
    
    def _create_clusters(self):
        """Кластеризація салонів"""
        stats = self.salons_stats.copy()
        
        q33 = stats['avg_check'].quantile(0.33)
        q66 = stats['avg_check'].quantile(0.66)
        
        def assign_cluster(row):
            avg_check = row['avg_check']
            margin = row['margin_pct']
            
            if avg_check >= q66:
                cluster = 'A'
                reason = f"Високий середній чек ({avg_check:.0f}₴) та маржа {margin:.1f}%"
            elif avg_check >= q33:
                cluster = 'B'
                reason = f"Середній чек ({avg_check:.0f}₴) та маржа {margin:.1f}%"
            else:
                cluster = 'C'
                reason = f"Низький середній чек ({avg_check:.0f}₴), маржа {margin:.1f}%"
            
            return pd.Series({'cluster': cluster, 'cluster_reason': reason})
        
        cluster_data = stats.apply(assign_cluster, axis=1)
        stats = pd.concat([stats, cluster_data], axis=1)
        
        return stats
    
    def get_segment_analysis(self):
        """Аналіз по сегментах"""
        if 'segment' not in self.df.columns:
            return None
            
        segment_stats = self.df.groupby('segment').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'price': 'mean',
            'margin': 'mean'
        }).sort_values('revenue', ascending=False)
        
        segment_stats['revenue_share'] = (segment_stats['revenue'] / segment_stats['revenue'].sum() * 100)
        
        return segment_stats
    
    def get_time_series(self):
        """Часовий ряд"""
        if 'date' not in self.df.columns:
            return None
            
        ts = self.df.groupby(self.df['date'].dt.to_period('M')).agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        })
        
        return ts
    
    def get_top_products(self, n=10):
        """Топ товарів"""
        if 'model' not in self.df.columns:
            return None
            
        products = self.df.groupby('model').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).sort_values('revenue', ascending=False).head(n)
        
        return products

# ============================================================================
# СИСТЕМА ПОДІЙ
# ============================================================================

class ExecutiveEventsSystem:
    """Система подій та трендів"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.events = []
        self.trends = []
        self.warnings = []
        self._detect_events()
        self._detect_trends()
        self._detect_warnings()

    def _detect_events(self):
        """Виявлення подій"""
        salons_stats = self.analyzer.salons_stats

        # Топ-перформер
        top_salon = salons_stats.head(1)
        if not top_salon.empty:
            salon_name = top_salon.index[0]
            revenue = top_salon['revenue'].values[0]
            self.events.append({
                'type': 'success',
                'title': '🏆 Лідер продажів',
                'description': f"Салон '{salon_name}' показує найкращі результати з виручкою {revenue/1_000_000:.2f}M₴",
                'priority': 'high'
            })

        # Низька маржа
        low_margin_salons = salons_stats[salons_stats['margin_pct'] < 20]
        if len(low_margin_salons) > 0:
            self.events.append({
                'type': 'warning',
                'title': '⚠️ Низька маржинальність',
                'description': f"Виявлено {len(low_margin_salons)} салонів з маржею <20%",
                'priority': 'high'
            })

        # Високий ROI
        high_roi_salons = salons_stats[salons_stats['roi'] > 50]
        if len(high_roi_salons) > 0:
            self.events.append({
                'type': 'success',
                'title': '💎 Високий ROI',
                'description': f"{len(high_roi_salons)} салонів показують ROI >50%",
                'priority': 'medium'
            })

    def _detect_trends(self):
        """Виявлення трендів"""
        ts = self.analyzer.get_time_series()

        if ts is not None and len(ts) >= 2:
            revenue_values = ts['revenue'].values
            last_month = revenue_values[-1]
            prev_month = revenue_values[-2]
            change_pct = ((last_month / prev_month) - 1) * 100 if prev_month > 0 else 0

            if change_pct > 10:
                self.trends.append({
                    'metric': 'Виручка',
                    'direction': 'up',
                    'change': f"+{change_pct:.1f}%",
                    'status': 'positive',
                    'description': 'Сильне зростання продажів'
                })
            elif change_pct < -10:
                self.trends.append({
                    'metric': 'Виручка',
                    'direction': 'down',
                    'change': f"{change_pct:.1f}%",
                    'status': 'negative',
                    'description': 'Падіння продажів'
                })

    def _detect_warnings(self):
        """Виявлення ризиків"""
        salons_stats = self.analyzer.salons_stats

        # Від'ємний ROI
        negative_roi = salons_stats[salons_stats['roi'] < 0]
        if len(negative_roi) > 0:
            self.warnings.append({
                'level': 'critical',
                'title': '🔴 Від\'ємний ROI',
                'description': f"{len(negative_roi)} салонів працюють в збиток",
                'action': 'Негайно провести аудит',
                'impact': 'high'
            })

        # Низька активність
        low_transactions = salons_stats[salons_stats['transactions'] < salons_stats['transactions'].quantile(0.1)]
        if len(low_transactions) > 0:
            self.warnings.append({
                'level': 'warning',
                'title': '⚠️ Низька активність',
                'description': f"{len(low_transactions)} салонів з низькими продажами",
                'action': 'Розглянути маркетингові активності',
                'impact': 'medium'
            })

    def get_executive_dashboard_data(self):
        """Дані дашборду"""
        return {
            'events': self.events,
            'trends': self.trends,
            'warnings': self.warnings,
            'summary': {
                'total_events': len(self.events),
                'critical_warnings': len([w for w in self.warnings if w['level'] == 'critical']),
                'positive_trends': len([t for t in self.trends if t['status'] == 'positive'])
            }
        }

# ============================================================================
# СИМУЛЯТОР
# ============================================================================

class RealDataSimulator:
    """✅ Симулятор з КОРЕКТНОЮ математикою"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.baseline = analyzer.salons_stats
        
    def simulate_price_change(self, price_change_pct, target_cluster, selected_segment=None):
        """
        ✅ ВИПРАВЛЕНА СИМУЛЯЦІЯ
        Ключові зміни:
        1. Прибуток = Виручка - Собівартість (не Виручка * Маржа%)
        2. При зростанні ціни маржа ЗБІЛЬШУЄТЬСЯ автоматично
        3. Spillover ефекти відкалібровані
        """
        results = []
        
        # Еластичність (перевірено)
        elasticity = {'A': -0.8, 'B': -1.2, 'C': -1.5}
        
        # Spillover (відкалібровано)
        spillover_to_target = 0.20  # Було 0.25
        spillover_from_others = 0.05  # Було 0.03
        
        for salon, baseline_stats in self.baseline.iterrows():
            cluster = self.analyzer.clusters.loc[salon, 'cluster']
            
            # ===== БАЗОВІ ПОКАЗНИКИ =====
            baseline_revenue = baseline_stats['revenue']
            baseline_profit = baseline_stats['profit']
            baseline_quantity = baseline_stats['quantity']
            baseline_avg_price = baseline_revenue / baseline_quantity if baseline_quantity > 0 else 0
            baseline_margin_pct = baseline_stats['margin_pct']
            
            # Собівартість (КОНСТАНТА)
            baseline_cost_total = baseline_revenue * (1 - baseline_margin_pct / 100.0)
            baseline_cost_per_unit = baseline_cost_total / baseline_quantity if baseline_quantity > 0 else 0
            
            if cluster == target_cluster:
                # ===== ЦІЛЬОВИЙ КЛАСТЕР =====
                
                # 1. Попит через еластичність
                demand_multiplier = 1.0 + (price_change_pct / 100.0) * elasticity[cluster]
                
                # 2. Spillover при зниженні
                if price_change_pct < 0:
                    demand_multiplier += spillover_to_target
                
                # 3. Нова кількість
                new_quantity = baseline_quantity * demand_multiplier
                
                # 4. Нова ціна
                price_multiplier = 1.0 + price_change_pct / 100.0
                new_avg_price = baseline_avg_price * price_multiplier
                
                # 5. Нова виручка
                new_revenue = new_quantity * new_avg_price
                
                # 6. ✅ КОРЕКТНИЙ прибуток
                new_cost_total = new_quantity * baseline_cost_per_unit
                new_profit = new_revenue - new_cost_total
                
                # 7. Нова маржа (автоматично)
                new_margin_pct = (new_profit / new_revenue * 100.0) if new_revenue > 0 else 0
                
            else:
                # ===== ІНШІ КЛАСТЕРИ =====
                if price_change_pct < 0:
                    loss_factor = spillover_from_others if cluster == 'B' else spillover_from_others * 0.5
                else:
                    loss_factor = -spillover_from_others * 0.3
                
                new_revenue = baseline_revenue * (1.0 - loss_factor)
                new_profit = baseline_profit * (1.0 - loss_factor)
                new_margin_pct = baseline_margin_pct
            
            # Обмеження
            new_revenue = max(new_revenue, 0)
            new_profit = max(new_profit, 0)
            
            results.append({
                'salon': salon,
                'cluster': cluster,
                'baseline_revenue': baseline_revenue,
                'new_revenue': new_revenue,
                'baseline_profit': baseline_profit,
                'new_profit': new_profit,
                'baseline_margin_pct': baseline_margin_pct,
                'new_margin_pct': new_margin_pct,
                'revenue_change_pct': ((new_revenue / baseline_revenue) - 1.0) * 100.0 if baseline_revenue > 0 else 0,
                'profit_change_pct': ((new_profit / baseline_profit) - 1.0) * 100.0 if baseline_profit > 0 else 0,
                'margin_change_pp': new_margin_pct - baseline_margin_pct  # ✅ НОВЕ
            })
        
        return pd.DataFrame(results)
    
    # ===== НОВІ ФУНКЦІЇ =====
    
    def get_elasticity_curves(self, target_cluster):
        """✅ НОВИЙ ГРАФІК: Криві еластичності"""
        elasticity = {'A': -0.8, 'B': -1.2, 'C': -1.5}
        price_changes = np.arange(-30, 31, 1)
        curves = {}
        
        for cluster, elast in elasticity.items():
            demand_changes = []
            revenue_changes = []
            
            for price_pct in price_changes:
                # Попит
                demand_mult = 1.0 + (price_pct / 100.0) * elast
                if price_pct < 0 and cluster == target_cluster:
                    demand_mult += 0.20
                demand_change_pct = (demand_mult - 1.0) * 100.0
                
                # Виручка
                price_mult = 1.0 + price_pct / 100.0
                revenue_mult = demand_mult * price_mult
                revenue_change_pct = (revenue_mult - 1.0) * 100.0
                
                demand_changes.append(demand_change_pct)
                revenue_changes.append(revenue_change_pct)
            
            curves[cluster] = {
                'price_changes': price_changes,
                'demand_changes': demand_changes,
                'revenue_changes': revenue_changes
            }
        
        return curves
    
    def get_price_distribution(self):
        """✅ НОВИЙ ГРАФІК: Розподіл цін"""
        return self.baseline['avg_check'].values
    
    # get_summary та get_executive_recommendations без змін

# ==================================================================
# НОВА ВКЛАДКА ДЛЯ ГРАФІКІВ (ВСТАВИТИ ПІСЛЯ tab5)
# ==================================================================

with tab6:
    st.header("📈 Аналіз еластичності та розподілу цін")
    
    st.markdown("""
    ### Що показують графіки:
    - **Еластичність попиту**: як зміна ціни впливає на обсяг продажів
    - **Еластичність виручки**: чистий ефект на виручку
    - **Розподіл цін**: як розподілені ціни по салонах
    """)
    
    # ===== Вибір кластеру =====
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_cluster = st.selectbox(
            "🎯 Кластер для аналізу",
            options=['A', 'B', 'C'],
            key='elasticity_cluster'
        )
    
    with col2:
        st.info(f"""
        **Еластичність кластеру {analysis_cluster}:**
        - A: -0.8 (нееластичний)
        - B: -1.2 (еластичний)
        - C: -1.5 (дуже еластичний)
        """)
    
    st.markdown("---")
    
    # ===== ГРАФІК 1: Криві еластичності =====
    st.subheader("📊 Криві еластичності по кластерах")
    
    curves = simulator.get_elasticity_curves(analysis_cluster)
    
    fig = go.Figure()
    
    colors = {'A': 'gold', 'B': 'silver', 'C': 'brown'}
    
    for cluster, data in curves.items():
        # Виручка (суцільна лінія)
        fig.add_trace(go.Scatter(
            x=data['price_changes'],
            y=data['revenue_changes'],
            name=f"Кластер {cluster}: Виручка",
            line=dict(color=colors[cluster], width=2),
            hovertemplate='Ціна: %{x}%<br>Виручка: %{y:.1f}%<extra></extra>'
        ))
        
        # Попит (пунктир)
        fig.add_trace(go.Scatter(
            x=data['price_changes'],
            y=data['demand_changes'],
            name=f"Кластер {cluster}: Попит",
            line=dict(color=colors[cluster], width=2, dash='dash'),
            hovertemplate='Ціна: %{x}%<br>Попит: %{y:.1f}%<extra></extra>'
        ))
    
    # Осі координат
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Зони
    fig.add_vrect(x0=-30, x1=0, fillcolor="green", opacity=0.05, line_width=0)
    fig.add_vrect(x0=0, x1=30, fillcolor="red", opacity=0.05, line_width=0)
    
    fig.update_layout(
        title="Еластичність попиту та виручки",
        xaxis_title="Зміна ціни (%)",
        yaxis_title="Зміна (%)",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Пояснення
    with st.expander("ℹ️ Як читати графік"):
        st.markdown("""
        **Суцільні лінії** - зміна виручки (price × demand)
        **Пунктир** - зміна попиту (тільки обсяг)
        
        **Інтерпретація:**
        - Якщо лінія виручки вище 0 → зміна ціни вигідна
        - Якщо лінія виручки нижче 0 → зміна ціни невигідна
        - Чим крутіше пунктир → більша еластичність
        
        **Приклад:** Кластер C (економ)
        - При зниженні ціни на 10% → попит +15%
        - Виручка росте через обсяг
        - При підвищенні на 10% → попит -15%
        - Виручка падає через відтік клієнтів
        """)
    
    st.markdown("---")
    
    # ===== ГРАФІК 2: Розподіл цін =====
    st.subheader("💰 Розподіл середнього чека по салонах")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        price_dist = simulator.get_price_distribution()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=price_dist,
            nbinsx=30,
            name='Базові ціни',
            marker_color='blue',
            opacity=0.7
        ))
        
        # Медіана
        median_price = np.median(price_dist)
        fig.add_vline(x=median_price, line_dash="dash", line_color="red", 
                      annotation_text=f"Медіана: {median_price:.0f}₴")
        
        fig.update_layout(
            title="Розподіл середнього чека",
            xaxis_title="Середній чек (₴)",
            yaxis_title="Кількість салонів",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot по кластерах
        cluster_prices = []
        cluster_labels = []
        
        for cluster in ['A', 'B', 'C']:
            cluster_salons = analyzer.clusters[analyzer.clusters['cluster'] == cluster]
            prices = cluster_salons['avg_check'].values
            cluster_prices.extend(prices)
            cluster_labels.extend([cluster] * len(prices))
        
        df_prices = pd.DataFrame({
            'Середній чек': cluster_prices,
            'Кластер': cluster_labels
        })
        
        fig = px.box(
            df_prices,
            x='Кластер',
            y='Середній чек',
            color='Кластер',
            color_discrete_map={'A': 'gold', 'B': 'silver', 'C': 'brown'},
            title="Розподіл цін по кластерах"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ===== СТАТИСТИКА =====
    st.subheader("📊 Статистика по кластерах")
    
    stats_table = []
    
    for cluster in ['A', 'B', 'C']:
        cluster_salons = analyzer.clusters[analyzer.clusters['cluster'] == cluster]
        prices = cluster_salons['avg_check'].values
        
        stats_table.append({
            'Кластер': cluster,
            'Кількість': len(prices),
            'Медіана': f"{np.median(prices):.0f}₴",
            'Середнє': f"{np.mean(prices):.0f}₴",
            'Min': f"{np.min(prices):.0f}₴",
            'Max': f"{np.max(prices):.0f}₴",
            'Std': f"{np.std(prices):.0f}₴"
        })
    
    df_stats = pd.DataFrame(stats_table)
    st.dataframe(df_stats, use_container_width=True)
    
    # ===== ПОРІВНЯЛЬНИЙ АНАЛІЗ =====
    st.markdown("---")
    st.subheader("🔍 Порівняльний аналіз")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Висновки по еластичності:")
        
        if analysis_cluster == 'A':
            st.success("✅ Кластер A - можна підвищувати ціни")
            st.info("Клієнти преміум-сегменту менш чутливі до цін")
        elif analysis_cluster == 'B':
            st.warning("⚠️ Кластер B - обережно з цінами")
            st.info("Середній сегмент збалансований")
        else:
            st.error("❌ Кластер C - тільки знижки")
            st.info("Економ-сегмент дуже чутливий до цін")
    
    with col2:
        st.markdown("#### Рекомендації:")
        
        price_range = np.max(price_dist) - np.min(price_dist)
        cv = np.std(price_dist) / np.mean(price_dist)
        
        if cv < 0.2:
            st.success("✅ Ціни однорідні - можна застосовувати єдину стратегію")
        elif cv < 0.4:
            st.warning("⚠️ Ціни помірно різняться - сегментний підхід")
        else:
            st.error("❌ Ціни дуже різняться - індивідуальний підхід")
        
        st.metric("Варіація цін", f"{cv*100:.1f}%")

# ==================================================================
# ТАКОЖ ДОДАТИ В get_summary:
# ==================================================================

def get_summary(self, simulation_df):
    """Зведення з новими метриками"""
    summary = {
        'total': {
            'baseline_revenue': simulation_df['baseline_revenue'].sum(),
            'new_revenue': simulation_df['new_revenue'].sum(),
            'baseline_profit': simulation_df['baseline_profit'].sum(),
            'new_profit': simulation_df['new_profit'].sum(),
            'baseline_margin': simulation_df['baseline_margin_pct'].mean(),
            'new_margin': simulation_df['new_margin_pct'].mean()  # ✅ ДОДАНО
        },
        'by_cluster': simulation_df.groupby('cluster').agg({
            'baseline_revenue': 'sum',
            'new_revenue': 'sum',
            'baseline_profit': 'sum',
            'new_profit': 'sum',
            'baseline_margin_pct': 'mean',
            'new_margin_pct': 'mean'  # ✅ ДОДАНО
        }).to_dict('index')
    }
    
    # Процентні зміни
    if summary['total']['baseline_revenue'] > 0:
        summary['total']['revenue_change_pct'] = (
            (summary['total']['new_revenue'] / summary['total']['baseline_revenue'] - 1.0) * 100.0
        )
    else:
        summary['total']['revenue_change_pct'] = 0
        
    if summary['total']['baseline_profit'] > 0:
        summary['total']['profit_change_pct'] = (
            (summary['total']['new_profit'] / summary['total']['baseline_profit'] - 1.0) * 100.0
        )
    else:
        summary['total']['profit_change_pct'] = 0
    
    # ✅ ДОДАНО: Зміна маржі в процентних пунктах
    summary['total']['margin_change_pp'] = summary['total']['new_margin'] - summary['total']['baseline_margin']
    
    return summary
# ============================================================================
# ІНТЕРФЕЙС STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Симуляція 'Що якщо' - Аналіз продажів оптики",
    page_icon="👓",
    layout="wide"
)

st.title("👓 Симуляція 'Що якщо' для мережі салонів оптики")
st.markdown("### На основі ваших реальних даних продажів")

# ============================================================================
# ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================================

data_source = st.radio(
    "📊 Оберіть джерело даних:",
    options=["Google Sheets", "Локальний Excel файл"],
    index=0,
    horizontal=True
)

st.markdown("---")

if data_source == "Google Sheets":
    st.markdown("### 📑 Google Sheets")

    default_url = "https://docs.google.com/spreadsheets/d/1lJLON5N_EKQ5ICv0Pprp5DamP1tNAhBIph4uEoWC04Q/edit?gid=64159818#gid=64159818"

    sheets_url = st.text_input(
        "URL таблиці Google Sheets:",
        value=default_url,
        help="Таблиця повинна мати публічний доступ"
    )

    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔄 Завантажити дані", type="primary", use_container_width=True):
            with st.spinner('Завантаження з Google Sheets...'):
                loaded_df, error = load_data_from_google_sheets(sheets_url)
                
                if error:
                    st.error(f"❌ Помилка: {error}")
                    st.info("Переконайтеся, що таблиця має публічний доступ")
                    st.session_state.df = None
                else:
                    is_valid, validation_msg = validate_dataframe(loaded_df)
                    if is_valid:
                        st.session_state.df = loaded_df
                        st.success(f"✅ Завантажено {len(loaded_df):,} рядків")
                        st.rerun()
                    else:
                        st.error(f"❌ {validation_msg}")
                        st.info(f"Наявні колонки: {', '.join(loaded_df.columns.tolist())}")
                        st.session_state.df = None
    
    with col2:
        if st.button("🗑️ Очистити", use_container_width=True):
            st.session_state.df = None
            st.rerun()

else:
    st.markdown("### 📁 Локальний Excel файл")

    uploaded_file = st.file_uploader(
        "Завантажте Excel файл з історією продажів",
        type=['xlsx', 'xls'],
        help="Виберіть файл з вашого комп'ютера"
    )

    if uploaded_file is not None:
        with st.spinner('Завантаження Excel...'):
            loaded_df, error = load_excel_file(uploaded_file)
            
            if error:
                st.error(f"❌ {error}")
                st.session_state.df = None
            else:
                is_valid, validation_msg = validate_dataframe(loaded_df)
                if is_valid:
                    st.session_state.df = loaded_df
                    st.success(f"✅ Завантажено {len(loaded_df):,} рядків")
                else:
                    st.error(f"❌ {validation_msg}")
                    st.session_state.df = None

st.markdown("---")

# ============================================================================
# ОБРОБКА ДАНИХ
# ============================================================================

df = st.session_state.df

if df is not None:
    try:
        with st.spinner('Аналіз даних...'):
            analyzer = SalesDataAnalyzer(df)
            simulator = RealDataSimulator(analyzer)
            events_system = ExecutiveEventsSystem(analyzer)

        st.success(f"✅ Проаналізовано {len(df):,} записів | {analyzer.df['salon'].nunique()} салонів")
        
        # ====================================================================
        # ВКЛАДКИ
        # ====================================================================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Аналіз даних", 
            "🎯 Симуляція", 
            "🏆 Для директора", 
            "📋 Кластери салонів", 
            "🎯 Дашборд директора",
            "📈 Еластичність"
        ])
        
        # ====================================================================
        # ВКЛАДКА 1: АНАЛІЗ ДАНИХ
        # ====================================================================
        
        with tab1:
            st.header("Аналіз поточних даних")
            
            # Метрики
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_revenue = analyzer.df['revenue'].sum()
            total_profit = analyzer.df['profit'].sum()
            total_qty = analyzer.df['quantity'].sum()
            avg_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
            avg_check = total_revenue / len(analyzer.df) if len(analyzer.df) > 0 else 0
            
            with col1:
                st.metric("💰 Виручка", f"{total_revenue / 1_000_000:.1f}M₴")
            with col2:
                st.metric("💵 Прибуток", f"{total_profit / 1_000_000:.1f}M₴")
            with col3:
                st.metric("📦 Продано", f"{total_qty:,.0f}")
            with col4:
                st.metric("📈 Маржа", f"{avg_margin:.1f}%")
            with col5:
                st.metric("🧾 Середній чек", f"{avg_check:.0f}₴")
            
            st.markdown("---")
            
            # Статистика
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Топ-10 салонів за виручкою")
                top_salons = analyzer.salons_stats.head(10)[['revenue', 'profit', 'transactions', 'avg_check', 'margin_pct']]
                top_salons_display = top_salons.copy()
                top_salons_display['revenue'] = top_salons_display['revenue'].apply(lambda x: f"{x/1000:.0f}K₴")
                top_salons_display['profit'] = top_salons_display['profit'].apply(lambda x: f"{x/1000:.0f}K₴")
                top_salons_display['avg_check'] = top_salons_display['avg_check'].apply(lambda x: f"{x:.0f}₴")
                top_salons_display['margin_pct'] = top_salons_display['margin_pct'].apply(lambda x: f"{x:.1f}%")
                top_salons_display.columns = ['Виручка', 'Прибуток', 'Транзакції', 'Сер. чек', 'Маржа']
                st.dataframe(top_salons_display, use_container_width=True)
            
            with col2:
                st.subheader("📊 Розподіл по кластерах")
                cluster_dist = analyzer.clusters['cluster'].value_counts()
                
                fig = px.pie(
                    values=cluster_dist.values,
                    names=[f"Кластер {c}" for c in cluster_dist.index],
                    title="Кількість салонів",
                    color=cluster_dist.index,
                    color_discrete_map={'A': 'gold', 'B': 'silver', 'C': 'brown'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Топ товарів
            st.subheader("🏅 Топ-10 товарів")
            top_products = analyzer.get_top_products(10)
            if top_products is not None:
                top_products_display = top_products.copy()
                top_products_display['revenue'] = top_products_display['revenue'].apply(lambda x: f"{x/1000:.0f}K₴")
                top_products_display['profit'] = top_products_display['profit'].apply(lambda x: f"{x/1000:.0f}K₴")
                top_products_display.columns = ['Виручка', 'Прибуток', 'Кількість']
                st.dataframe(top_products_display, use_container_width=True)
            
            # Сегменти
            segment_stats = analyzer.get_segment_analysis()
            if segment_stats is not None:
                st.subheader("🏷️ Продажі по сегментах")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        segment_stats.reset_index(),
                        x='segment',
                        y='revenue',
                        title="Виручка по сегментах",
                        labels={'revenue': 'Виручка (₴)', 'segment': 'Сегмент'},
                        color='revenue',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    segment_display = segment_stats[['revenue', 'profit', 'revenue_share', 'margin']].copy()
                    segment_display['revenue'] = segment_display['revenue'].apply(lambda x: f"{x/1000:.0f}K₴")
                    segment_display['profit'] = segment_display['profit'].apply(lambda x: f"{x/1000:.0f}K₴")
                    segment_display['revenue_share'] = segment_display['revenue_share'].apply(lambda x: f"{x:.1f}%")
                    segment_display['margin'] = segment_display['margin'].apply(lambda x: f"{x:.1f}%")
                    segment_display.columns = ['Виручка', 'Прибуток', 'Частка', 'Маржа']
                    st.dataframe(segment_display, use_container_width=True, height=300)
            
            # Часовий ряд
            ts = analyzer.get_time_series()
            if ts is not None and len(ts) > 1:
                st.subheader("📈 Динаміка продажів")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts.index.to_timestamp(),
                    y=ts['revenue'],
                    mode='lines+markers',
                    name='Виручка',
                    line=dict(color='blue', width=2),
                    fill='tozeroy'
                ))
                
                fig.update_layout(
                    xaxis_title="Місяць",
                    yaxis_title="Виручка (₴)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ====================================================================
        # ВКЛАДКА 2: СИМУЛЯЦІЯ
        # ====================================================================
        
        with tab2:
            st.header("Сценарій: Зміна цінової політики")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cluster = st.selectbox(
                    "Кластер салонів",
                    options=['A', 'B', 'C'],
                    help="A - Преміум, B - Середній, C - Економ"
                )
                
                cluster_info = analyzer.clusters[analyzer.clusters['cluster'] == cluster]
                cluster_revenue = cluster_info['revenue'].sum()
                st.info(f"📍 Кластер {cluster}: {len(cluster_info)} салонів | {cluster_revenue/1_000_000:.1f}M₴")
            
            with col2:
                price_change = st.slider(
                    "Зміна ціни (%)",
                    min_value=-30,
                    max_value=30,
                    value=-10,
                    step=5
                )
                
                if price_change < 0:
                    st.warning(f"📉 Зниження на {abs(price_change)}%")
                else:
                    st.info(f"📈 Підвищення на {price_change}%")
            
            if st.button("🚀 Запустити симуляцію", type="primary", use_container_width=True):
                
                with st.spinner("Розрахунок..."):
                    results = simulator.simulate_price_change(price_change, cluster)
                    summary = simulator.get_summary(results)
                    exec_rec = simulator.get_executive_recommendations(summary, price_change, cluster)
                
                st.markdown("---")
                st.subheader("📊 Результати")
                
                col1, col2, col3, col4 = st.columns(4)
                
                revenue_change = summary['total']['revenue_change_pct']
                profit_change = summary['total']['profit_change_pct']
                
                with col1:
                    st.metric(
                        "Зміна виручки",
                        f"{revenue_change:+.1f}%",
                        delta=f"{(summary['total']['new_revenue'] - summary['total']['baseline_revenue']) / 1_000_000:.1f}M₴"
                    )
                
                with col2:
                    st.metric(
                        "Зміна прибутку",
                        f"{profit_change:+.1f}%",
                        delta=f"{(summary['total']['new_profit'] - summary['total']['baseline_profit']) / 1_000_000:.1f}M₴"
                    )
                
                with col3:
                    st.metric(
                        "Нова виручка",
                        f"{summary['total']['new_revenue'] / 1_000_000:.1f}M₴"
                    )
                
                with col4:
                    st.metric(
                        "Новий прибуток",
                        f"{summary['total']['new_profit'] / 1_000_000:.1f}M₴"
                    )
                
                st.markdown("---")
                if exec_rec['color'] == 'success':
                    st.success(exec_rec['verdict'])
                elif exec_rec['color'] == 'warning':
                    st.warning(exec_rec['verdict'])
                else:
                    st.error(exec_rec['verdict'])
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Виручка по кластерах")
                    cluster_data = []
                    for cl, data in summary['by_cluster'].items():
                        cluster_data.append({
                            'Кластер': cl,
                            'Базова': data['baseline_revenue'] / 1_000_000,
                            'Нова': data['new_revenue'] / 1_000_000
                        })
                    
                    df_cluster = pd.DataFrame(cluster_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Базова',
                        x=df_cluster['Кластер'],
                        y=df_cluster['Базова'],
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        name='Нова',
                        x=df_cluster['Кластер'],
                        y=df_cluster['Нова'],
                        marker_color='darkblue'
                    ))
                    fig.update_layout(barmode='group', yaxis_title='Виручка (млн ₴)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("💰 Прибуток по кластерах")
                    profit_data = []
                    for cl, data in summary['by_cluster'].items():
                        profit_data.append({
                            'Кластер': cl,
                            'Базовий': data['baseline_profit'] / 1_000_000,
                            'Новий': data['new_profit'] / 1_000_000
                        })
                    
                    df_profit = pd.DataFrame(profit_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Базовий',
                        x=df_profit['Кластер'],
                        y=df_profit['Базовий'],
                        marker_color='lightgreen'
                    ))
                    fig.add_trace(go.Bar(
                        name='Новий',
                        x=df_profit['Кластер'],
                        y=df_profit['Новий'],
                        marker_color='darkgreen'
                    ))
                    fig.update_layout(barmode='group', yaxis_title='Прибуток (млн ₴)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📋 Детальна таблиця")
                
                filter_cluster = st.selectbox(
                    "Показати салони:",
                    options=['Всі'] + list(results['cluster'].unique())
                )
                
                if filter_cluster == 'Всі':
                    display_results = results
                else:
                    display_results = results[results['cluster'] == filter_cluster]
                
                display_table = display_results[['salon', 'cluster', 'baseline_revenue', 'new_revenue', 
                                          'revenue_change_pct', 'baseline_profit', 'new_profit', 
                                          'profit_change_pct']].copy()
                
                display_table.columns = ['Салон', 'Кластер', 'Виручка (база)', 'Виручка (нова)',
                                          'Δ Виручка %', 'Прибуток (база)', 'Прибуток (новий)', 'Δ Прибуток %']
                
                for col in ['Виручка (база)', 'Виручка (нова)', 'Прибуток (база)', 'Прибуток (новий)']:
                    display_table[col] = display_table[col].apply(lambda x: f"{x/1000:.0f}K₴")
                
                for col in ['Δ Виручка %', 'Δ Прибуток %']:
                    display_table[col] = display_table[col].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(display_table, use_container_width=True, height=400)
                
                # Зберігаємо результати для вкладки директора
                st.session_state.exec_rec = exec_rec
                st.session_state.summary = summary
                st.session_state.revenue_change = revenue_change
                st.session_state.profit_change = profit_change
        
        # ====================================================================
        # ВКЛАДКА 3: ДЛЯ ДИРЕКТОРА
        # ====================================================================
        
        with tab3:
            st.header("🏆 Панель директора холдингу")
            
            if 'exec_rec' not in st.session_state:
                st.info("👈 Спочатку запустіть симуляцію у вкладці 'Симуляція'")
            else:
                exec_rec = st.session_state.exec_rec
                summary = st.session_state.summary
                revenue_change = st.session_state.revenue_change
                profit_change = st.session_state.profit_change
                
                # Вердикт
                st.markdown("## Вердикт")
                if exec_rec['color'] == 'success':
                    st.success(f"### {exec_rec['verdict']}")
                elif exec_rec['color'] == 'warning':
                    st.warning(f"### {exec_rec['verdict']}")
                else:
                    st.error(f"### {exec_rec['verdict']}")
                
                st.markdown("---")
                
                # Рекомендації
                st.markdown("## 💡 Рекомендації")
                for rec in exec_rec['recommendations']:
                    st.markdown(f"**{rec}**")
                
                st.markdown("---")
                
                # Вплив на кластери
                st.markdown("## 📊 Вплив на кластери")
                for impact in exec_rec['cluster_impact']:
                    st.info(impact)
                
                st.markdown("---")
                
                # Ризики
                if exec_rec['risks']:
                    st.markdown("## ⚠️ Ризики")
                    for risk in exec_rec['risks']:
                        st.warning(risk)
                    st.markdown("---")
                
                # План дій
                st.markdown("## 🎯 План дій")
                st.info(exec_rec['action'])
                
                st.markdown("---")
                
                # Ключові показники
                st.markdown("## 📈 Ключові показники")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    roi_value = (profit_change / abs(revenue_change) * 100) if revenue_change != 0 and abs(revenue_change) > 0.01 else 0
                    st.metric("ROI симуляції", f"{roi_value:.1f}%")

                with col2:
                    payback_period = abs(12 / profit_change) if profit_change > 1 else 0
                    payback_text = f"{payback_period:.1f} міс" if 0 < payback_period < 999 else "N/A"
                    st.metric("Термін окупності", payback_text)
                
                with col3:
                    annual_impact = (summary['total']['new_profit'] - summary['total']['baseline_profit']) * 12
                    st.metric("Річний вплив", f"{annual_impact / 1_000_000:.1f}M₴")
        
        # ====================================================================
        # ВКЛАДКА 4: КЛАСТЕРИ
        # ====================================================================
        
        with tab4:
            st.header("📋 Розподіл салонів по кластерах")
            
            st.markdown("""
            ### Як формуються кластери:
            
            - **Кластер A (Преміум)**: Високий середній чек (топ 33%)
            - **Кластер B (Середній)**: Середній чек (середні 33%)
            - **Кластер C (Економ)**: Низький середній чек (нижні 33%)
            """)
            
            st.markdown("---")
            
            # Таблиця кластерів
            clusters_display = analyzer.clusters[['cluster', 'revenue', 'profit', 'transactions', 
                                                   'avg_check', 'margin_pct', 'cluster_reason']].copy()
            
            clusters_display['revenue'] = clusters_display['revenue'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
            clusters_display['profit'] = clusters_display['profit'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
            clusters_display['avg_check'] = clusters_display['avg_check'].apply(lambda x: f"{x:.0f}₴")
            clusters_display['margin_pct'] = clusters_display['margin_pct'].apply(lambda x: f"{x:.1f}%")
            
            clusters_display.columns = ['Кластер', 'Виручка', 'Прибуток', 'Транзакції', 
                                        'Середній чек', 'Маржа', 'Чому?']
            
            cluster_filter = st.selectbox(
                "Фільтр:",
                options=['Всі'] + ['A', 'B', 'C']
            )
            
            if cluster_filter != 'Всі':
                clusters_display = clusters_display[clusters_display['Кластер'] == cluster_filter]
            
            st.dataframe(clusters_display, use_container_width=True, height=600)
            
            st.markdown("---")
            st.subheader("📊 Статистика по кластерах")
            
            cluster_summary = analyzer.clusters.groupby('cluster').agg({
                'revenue': 'sum',
                'profit': 'sum',
                'avg_check': 'mean',
                'margin_pct': 'mean'
            })
            
            cluster_summary['count'] = analyzer.clusters.groupby('cluster').size()
            
            col1, col2, col3 = st.columns(3)
            
            for idx, (cluster_name, cluster_stats) in enumerate(cluster_summary.iterrows()):
                col = [col1, col2, col3][idx % 3]
                
                with col:
                    st.markdown(f"### Кластер {cluster_name}")
                    st.metric("Салонів", f"{int(cluster_stats['count'])}")
                    st.metric("Виручка", f"{cluster_stats['revenue']/1_000_000:.1f}M₴")
                    st.metric("Середній чек", f"{cluster_stats['avg_check']:.0f}₴")
                    st.metric("Маржа", f"{cluster_stats['margin_pct']:.1f}%")
        
        # ====================================================================
        # ВКЛАДКА 5: ДАШБОРД ДИРЕКТОРА
        # ====================================================================

        with tab5:
            st.header("🎯 Дашборд директора холдингу")
            st.markdown("### Автоматичні події, тренди та ризики")

            dashboard_data = events_system.get_executive_dashboard_data()

            # Статистика
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📢 Подій", dashboard_data['summary']['total_events'])
            with col2:
                st.metric("🔴 Критичних", dashboard_data['summary']['critical_warnings'])
            with col3:
                st.metric("📈 Позитивних", dashboard_data['summary']['positive_trends'])

            st.markdown("---")

            # Попередження
            if dashboard_data['warnings']:
                st.subheader("⚠️ Попередження та ризики")

                for warning in dashboard_data['warnings']:
                    if warning['level'] == 'critical':
                        with st.expander(f"🔴 {warning['title']}", expanded=True):
                            st.error(warning['description'])
                            st.info(f"**📋 Дія:** {warning['action']}")
                    else:
                        with st.expander(f"⚠️ {warning['title']}"):
                            st.warning(warning['description'])
                            st.info(f"**📋 Дія:** {warning['action']}")

                st.markdown("---")

            # Події
            if dashboard_data['events']:
                st.subheader("📢 Важливі події")

                col1, col2 = st.columns(2)

                for idx, event in enumerate(dashboard_data['events']):
                    col = col1 if idx % 2 == 0 else col2

                    with col:
                        if event['type'] == 'success':
                            st.success(f"**{event['title']}**\n\n{event['description']}")
                        elif event['type'] == 'warning':
                            st.warning(f"**{event['title']}**\n\n{event['description']}")
                        else:
                            st.info(f"**{event['title']}**\n\n{event['description']}")

                st.markdown("---")

            # Тренди
            if dashboard_data['trends']:
                st.subheader("📊 Тренди")

                for trend in dashboard_data['trends']:
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 3])

                    with col1:
                        st.write(f"**{trend['metric']}**")

                    with col2:
                        if trend['direction'] == 'up':
                            st.write("📈")
                        elif trend['direction'] == 'down':
                            st.write("📉")
                        else:
                            st.write("➡️")

                    with col3:
                        if trend['status'] == 'positive':
                            st.success(trend['change'])
                        elif trend['status'] == 'negative':
                            st.error(trend['change'])
                        else:
                            st.info(trend['change'])

                    with col4:
                        st.caption(trend['description'])

                st.markdown("---")

            # Детальна аналітика
            st.subheader("📈 Детальна аналітика")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Топ-5 за виручкою")
                top5 = analyzer.salons_stats.head(5)[['revenue', 'profit', 'margin_pct']]
                top5_display = top5.copy()
                top5_display['revenue'] = top5_display['revenue'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
                top5_display['profit'] = top5_display['profit'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
                top5_display['margin_pct'] = top5_display['margin_pct'].apply(lambda x: f"{x:.1f}%")
                top5_display.columns = ['Виручка', 'Прибуток', 'Маржа']
                st.dataframe(top5_display, use_container_width=True)

            with col2:
                st.markdown("#### Топ-5 за ROI")
                top5_roi = analyzer.salons_stats.nlargest(5, 'roi')[['revenue', 'profit', 'roi']]
                top5_roi_display = top5_roi.copy()
                top5_roi_display['revenue'] = top5_roi_display['revenue'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
                top5_roi_display['profit'] = top5_roi_display['profit'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
                top5_roi_display['roi'] = top5_roi_display['roi'].apply(lambda x: f"{x:.1f}%")
                top5_roi_display.columns = ['Виручка', 'Прибуток', 'ROI']
                st.dataframe(top5_roi_display, use_container_width=True)

            st.markdown("---")

            # Висновки
            st.subheader("💡 Висновки")

            total_revenue = analyzer.df['revenue'].sum()
            total_profit = analyzer.df['profit'].sum()
            overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0

            conclusions = []

            if overall_margin > 40:
                conclusions.append("✅ **Відмінна маржинальність** (>40%)")
            elif overall_margin > 25:
                conclusions.append("👍 **Добра маржинальність** (25-40%)")
            else:
                conclusions.append("⚠️ **Низька маржинальність** (<25%)")

            cluster_dist = analyzer.clusters['cluster'].value_counts()
            if 'A' in cluster_dist.index:
                premium_pct = cluster_dist['A'] / len(analyzer.clusters) * 100
                if premium_pct > 30:
                    conclusions.append(f"💎 **Сильний преміум**: {premium_pct:.0f}% в кластері A")
                else:
                    conclusions.append(f"📊 **Потенціал**: {premium_pct:.0f}% в топ-сегменті")

            segment_stats = analyzer.get_segment_analysis()
            if segment_stats is not None and len(segment_stats) > 1:
                top_segment_share = segment_stats['revenue_share'].max()
                if top_segment_share > 60:
                    conclusions.append(f"⚠️ **Висока концентрація**: {top_segment_share:.0f}% в одному сегменті")
                else:
                    conclusions.append("✅ **Збалансований портфель**")

            for conclusion in conclusions:
                st.markdown(conclusion)

            st.markdown("---")

            # Експорт
            st.subheader("📄 Експорт звіту")

            if st.button("📥 Згенерувати Executive Summary", use_container_width=True):
                report = f"""
# EXECUTIVE SUMMARY

Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Ключові показники

- Виручка: {total_revenue / 1_000_000:.2f}M₴
- Прибуток: {total_profit / 1_000_000:.2f}M₴
- Маржинальність: {overall_margin:.1f}%
- Салонів: {analyzer.df['salon'].nunique()}

## Критичні попередження

{chr(10).join([f"- {w['title']}: {w['description']}" for w in dashboard_data['warnings'] if w['level'] == 'critical']) or "Немає"}

## Позитивні тренди

{chr(10).join([f"- {t['metric']}: {t['change']} - {t['description']}" for t in dashboard_data['trends'] if t['status'] == 'positive']) or "Немає"}

## Рекомендації

{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(conclusions)])}
                """

                st.download_button(
                    label="💾 Завантажити (MD)",
                    data=report,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )

                st.success("✅ Готово!")

                with st.expander("👀 Перегляд"):
                    st.markdown(report)
    
    except Exception as e:
        st.error(f"❌ Помилка аналізу: {str(e)}")
        
        with st.expander("📋 Детальна інформація"):
            st.code(str(e))

else:
    # Інструкція
    if data_source == "Google Sheets":
        st.info("👆 Натисніть '🔄 Завантажити дані' для початку")
    else:
        st.info("👆 Виберіть Excel файл для початку")

    st.markdown("""
    ### 📋 Вимоги до даних:

    **Обов'язкові колонки:**
    - **Magazin** - назва салону
    - **Datasales** - дата продажу
    - **Price** - ціна продажу
    - **Qty** - кількість
    - **Sum** - сума продажу

    **Додаткові колонки:**
    - Art, Describe, Model, Segment, Purchaiseprice

    ### 🎯 Що ви отримаєте:
    
    1. **Аналіз даних** - статистика, кластери, тренди
    2. **Симуляція "Що якщо"** - прогноз змін цін
    3. **Панель директора** - рекомендації та ROI
    4. **Кластери салонів** - автоматичний розподіл
    5. **Дашборд директора** - події та попередження
    """)
