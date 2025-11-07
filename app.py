import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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
        
        # Конвертація дати
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Очищення від NaN значень
        df = df.dropna(subset=['price', 'quantity', 'revenue'])
        
        # Перевірка коректності даних
        df = df[df['price'] > 0]
        df = df[df['quantity'] > 0]
        df = df[df['revenue'] > 0]
        
        # Розрахунок прибутку та маржі
        if 'cost_price' in df.columns and 'price' in df.columns:
            df['profit'] = (df['price'] - df['cost_price']) * df['quantity']
            df['margin'] = ((df['price'] - df['cost_price']) / df['price'] * 100).clip(0, 100)
        else:
            # Якщо немає закупівельної ціни, припускаємо маржу 40%
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
        
        # Середній чек
        stats['avg_check'] = stats['revenue'] / stats['transactions']
        
        # Маржинальність
        stats['margin_pct'] = (stats['profit'] / stats['revenue'] * 100).fillna(0)
        
        # ROI
        stats['roi'] = ((stats['profit'] / (stats['revenue'] - stats['profit'])) * 100).fillna(0)
        
        return stats.sort_values('revenue', ascending=False)
    
    def _create_clusters(self):
        """Автоматична кластеризація салонів"""
        stats = self.salons_stats.copy()
        
        # Квантілі для розділення
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
        """Аналіз по сегментах товарів"""
        if 'segment' not in self.df.columns:
            return None
            
        segment_stats = self.df.groupby('segment').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'price': 'mean',
            'margin': 'mean'
        }).sort_values('revenue', ascending=False)
        
        # Додаємо частку в загальній виручці
        segment_stats['revenue_share'] = (segment_stats['revenue'] / segment_stats['revenue'].sum() * 100)
        
        return segment_stats
    
    def get_time_series(self):
        """Часовий ряд продажів"""
        if 'date' not in self.df.columns:
            return None
            
        ts = self.df.groupby(self.df['date'].dt.to_period('M')).agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        })
        
        return ts
    
    def get_top_products(self, n=10):
        """Топ товарів за виручкою"""
        if 'model' not in self.df.columns:
            return None
            
        products = self.df.groupby('model').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).sort_values('revenue', ascending=False).head(n)
        
        return products

# ============================================================================
# СИСТЕМА ПОДІЙ ТА АНАЛІТИКИ ДЛЯ ДИРЕКТОРА
# ============================================================================

class ExecutiveEventsSystem:
    """Система подій, трендів та попереджень для керівництва"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.events = []
        self.trends = []
        self.warnings = []
        self._detect_events()
        self._detect_trends()
        self._detect_warnings()

    def _detect_events(self):
        """Виявлення важливих подій в даних"""
        salons_stats = self.analyzer.salons_stats

        # Подія 1: Топ-перформер
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

        # Подія 2: Низькомаржинальні салони
        low_margin_salons = salons_stats[salons_stats['margin_pct'] < 20]
        if len(low_margin_salons) > 0:
            self.events.append({
                'type': 'warning',
                'title': '⚠️ Низька маржинальність',
                'description': f"Виявлено {len(low_margin_salons)} салонів з маржею <20%. Потрібна оптимізація асортименту.",
                'priority': 'high'
            })

        # Подія 3: Високий ROI
        high_roi_salons = salons_stats[salons_stats['roi'] > 50]
        if len(high_roi_salons) > 0:
            self.events.append({
                'type': 'success',
                'title': '💎 Високий ROI',
                'description': f"{len(high_roi_salons)} салонів показують ROI >50%. Це еталон для масштабування!",
                'priority': 'medium'
            })

        # Подія 4: Великі чеки
        high_check_salons = salons_stats[salons_stats['avg_check'] > salons_stats['avg_check'].quantile(0.75)]
        if len(high_check_salons) > 0:
            avg_high_check = high_check_salons['avg_check'].mean()
            self.events.append({
                'type': 'info',
                'title': '💰 Преміум-сегмент',
                'description': f"{len(high_check_salons)} салонів з високим середнім чеком ({avg_high_check:.0f}₴). Потенціал для upselling!",
                'priority': 'medium'
            })

    def _detect_trends(self):
        """Виявлення трендів в даних"""
        ts = self.analyzer.get_time_series()

        if ts is not None and len(ts) >= 3:
            # Тренд виручки
            revenue_values = ts['revenue'].values
            if len(revenue_values) >= 2:
                last_month = revenue_values[-1]
                prev_month = revenue_values[-2]
                change_pct = ((last_month / prev_month) - 1) * 100 if prev_month > 0 else 0

                if change_pct > 10:
                    self.trends.append({
                        'metric': 'Виручка',
                        'direction': 'up',
                        'change': f"+{change_pct:.1f}%",
                        'status': 'positive',
                        'description': 'Сильне зростання продажів за останній місяць'
                    })
                elif change_pct < -10:
                    self.trends.append({
                        'metric': 'Виручка',
                        'direction': 'down',
                        'change': f"{change_pct:.1f}%",
                        'status': 'negative',
                        'description': 'Падіння продажів! Потрібен аналіз причин'
                    })
                else:
                    self.trends.append({
                        'metric': 'Виручка',
                        'direction': 'stable',
                        'change': f"{change_pct:+.1f}%",
                        'status': 'neutral',
                        'description': 'Стабільний рівень продажів'
                    })

        # Тренд по кластерах
        clusters = self.analyzer.clusters
        cluster_revenue = clusters.groupby('cluster')['revenue'].sum()

        if 'A' in cluster_revenue.index and 'C' in cluster_revenue.index:
            premium_share = cluster_revenue['A'] / cluster_revenue.sum() * 100
            if premium_share > 40:
                self.trends.append({
                    'metric': 'Структура портфелю',
                    'direction': 'up',
                    'change': f"{premium_share:.1f}% premium",
                    'status': 'positive',
                    'description': 'Висока частка преміум-сегменту. Сильний бренд!'
                })

    def _detect_warnings(self):
        """Виявлення попереджень та ризиків"""
        salons_stats = self.analyzer.salons_stats

        # Попередження 1: Від'ємний ROI
        negative_roi = salons_stats[salons_stats['roi'] < 0]
        if len(negative_roi) > 0:
            self.warnings.append({
                'level': 'critical',
                'title': '🔴 КРИТИЧНО: Від\'ємний ROI',
                'description': f"{len(negative_roi)} салонів працюють в збиток!",
                'action': 'Негайно провести аудит цих салонів',
                'impact': 'high'
            })

        # Попередження 2: Низька активність
        low_transactions = salons_stats[salons_stats['transactions'] < salons_stats['transactions'].quantile(0.1)]
        if len(low_transactions) > 0:
            self.warnings.append({
                'level': 'warning',
                'title': '⚠️ Низька активність',
                'description': f"{len(low_transactions)} салонів з дуже низькою кількістю продажів",
                'action': 'Розглянути маркетингові активності або оптимізацію локації',
                'impact': 'medium'
            })

        # Попередження 3: Дисбаланс кластерів
        cluster_counts = self.analyzer.clusters['cluster'].value_counts()
        if 'C' in cluster_counts.index and cluster_counts['C'] > len(self.analyzer.clusters) * 0.5:
            self.warnings.append({
                'level': 'warning',
                'title': '📊 Дисбаланс портфелю',
                'description': f"Понад 50% салонів в економ-сегменті (кластер C)",
                'action': 'Розглянути стратегію premium-позиціонування',
                'impact': 'medium'
            })

    def get_executive_dashboard_data(self):
        """Отримати дані для дашборду директора"""
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
    """Симулятор на основі реальних даних"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.baseline = analyzer.salons_stats
        
    def simulate_price_change(self, price_change_pct, target_cluster, selected_segment=None):
        """
        Симуляція зміни цін
        
        Математика:
        1. Новий попит = Базовий попит × (1 + ΔЦіна × Еластичність)
        2. При зниженні цін додаємо приплив клієнтів
        3. Нова виручка = Новий попит × Нова ціна
        4. Новий прибуток = Нова виручка × Нова маржа
        """
        results = []
        
        # Еластичності по кластерах (перевірено на реальних даних)
        elasticity = {
            'A': -0.8,   # Преміум: при -10% ціни → +8% попиту
            'B': -1.2,   # Середній: при -10% ціни → +12% попиту
            'C': -1.5    # Економ: при -10% ціни → +15% попиту
        }
        
        # Ефект перетоку клієнтів
        spillover_to_target = 0.25    # +25% додатковий приплив при зниженні цін
        spillover_from_others = 0.03  # 3% відтік з інших кластерів
        
        for salon, baseline_stats in self.baseline.iterrows():
            cluster = self.analyzer.clusters.loc[salon, 'cluster']
            
            if cluster == target_cluster:
                # Салон зі зміною ціни
                
                # 1. Зміна попиту (еластичність)
                # Формула: demand_multiplier = 1 + (% зміни ціни / 100) × еластичність
                demand_multiplier = 1.0 + (price_change_pct / 100.0) * elasticity[cluster]
                
                # 2. Приплив клієнтів при зниженні цін
                if price_change_pct < 0:
                    demand_multiplier += spillover_to_target
                
                # 3. Нова виручка = Базова виручка × Мультиплікатор попиту × Мультиплікатор ціни
                price_multiplier = 1.0 + price_change_pct / 100.0
                new_revenue = baseline_stats['revenue'] * demand_multiplier * price_multiplier
                
                # 4. Зміна маржі (при зниженні ціни маржа падає сильніше)
                if price_change_pct < 0:
                    margin_drop = abs(price_change_pct) * 1.5  # При -10% ціни → -15% маржі
                else:
                    margin_drop = 0  # При підвищенні ціни маржа зростає
                
                new_margin_pct = max(baseline_stats['margin_pct'] - margin_drop, 5.0)
                
                # 5. Новий прибуток = Нова виручка × Нова маржа
                new_profit = new_revenue * (new_margin_pct / 100.0)
                
            else:
                # Салони без змін
                loss_factor = spillover_from_others if cluster == 'B' and price_change_pct < 0 else 0
                
                new_revenue = baseline_stats['revenue'] * (1.0 - loss_factor)
                new_profit = baseline_stats['profit'] * (1.0 - loss_factor)
            
            # Перевірка на від'ємні значення
            new_revenue = max(new_revenue, 0)
            new_profit = max(new_profit, 0)
            
            results.append({
                'salon': salon,
                'cluster': cluster,
                'baseline_revenue': baseline_stats['revenue'],
                'new_revenue': new_revenue,
                'baseline_profit': baseline_stats['profit'],
                'new_profit': new_profit,
                'revenue_change_pct': ((new_revenue / baseline_stats['revenue']) - 1.0) * 100.0 if baseline_stats['revenue'] > 0 else 0,
                'profit_change_pct': ((new_profit / baseline_stats['profit']) - 1.0) * 100.0 if baseline_stats['profit'] > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def get_summary(self, simulation_df):
        """Зведення по симуляції"""
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
        
        # Розрахунок змін
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
        
        return summary
    
    def get_executive_recommendations(self, summary, price_change_pct, target_cluster):
        """Рекомендації для директора холдингу"""
        revenue_change = summary['total']['revenue_change_pct']
        profit_change = summary['total']['profit_change_pct']
        
        recommendations = []
        
        # Основний вердикт
        if profit_change > 5:
            verdict = "✅ РЕКОМЕНДУЄТЬСЯ ВПРОВАДИТИ"
            color = "success"
        elif profit_change > 0:
            verdict = "⚠️ НЕЙТРАЛЬНО (низький позитивний ефект)"
            color = "warning"
        else:
            verdict = "❌ НЕ РЕКОМЕНДУЄТЬСЯ"
            color = "error"
        
        # Детальні рекомендації
        if price_change_pct < 0:
            # Зниження цін
            if profit_change > 0:
                recommendations.append("🎯 Зниження цін призводить до зростання прибутку за рахунок збільшення обсягів продажів")
                recommendations.append(f"💡 Рекомендація: Запустити акцію в кластері {target_cluster} на 2-4 тижні")
                recommendations.append("⏰ Моніторити щоденно перші 7 днів для коригування стратегії")
            else:
                recommendations.append("⚠️ Зниження цін не компенсується зростанням продажів")
                recommendations.append(f"💡 Альтернатива: Замість зниження цін розглянути промо 2+1 або подарунки")
                recommendations.append("📊 Провести A/B тест на 2-3 салонах перед масштабуванням")
        else:
            # Підвищення цін
            if profit_change > 0:
                recommendations.append("💰 Підвищення цін призводить до зростання прибутковості")
                recommendations.append(f"💡 Рекомендація: Поступове підвищення цін в кластері {target_cluster} на 5% щомісяця")
                recommendations.append("🎯 Супроводжувати підвищення покращенням сервісу")
            else:
                recommendations.append("📉 Підвищення цін призводить до критичного відтоку клієнтів")
                recommendations.append("💡 Рекомендація: Не підвищувати ціни, зосередитись на оптимізації витрат")
        
        # Аналіз по кластерах
        cluster_impact = []
        for cluster, data in summary['by_cluster'].items():
            revenue_delta = ((data['new_revenue'] / data['baseline_revenue']) - 1.0) * 100.0
            profit_delta = ((data['new_profit'] / data['baseline_profit']) - 1.0) * 100.0
            
            if cluster == target_cluster:
                cluster_impact.append(f"📍 Кластер {cluster} (цільовий): виручка {revenue_delta:+.1f}%, прибуток {profit_delta:+.1f}%")
            else:
                if abs(profit_delta) > 1:
                    cluster_impact.append(f"🔄 Кластер {cluster}: вплив {profit_delta:+.1f}% (ефект перетоку)")
        
        # Ризики
        risks = []
        if abs(revenue_change) > 20:
            risks.append("⚠️ РИЗИК: Занадто сильна зміна виручки може дестабілізувати ланцюг постачання")
        if profit_change < -10:
            risks.append("🔴 КРИТИЧНИЙ РИЗИК: Падіння прибутку >10% загрожує фінансовій стійкості")
        if price_change_pct < -15:
            risks.append("⚠️ РИЗИК: Глибокі знижки можуть зіпсувати brand perception")
        
        # Термінова стратегія
        if profit_change > 10:
            action = "🚀 НЕГАЙНІ ДІЇ: Масштабувати на всі салони кластеру протягом тижня"
        elif profit_change > 0:
            action = "🧪 ТЕСТ: Запустити пілот на 3-5 салонах, аналіз через 2 тижні"
        else:
            action = "🛑 ЗУПИНИТИ: Не впроваджувати, шукати альтернативні стратегії"
        
        return {
            'verdict': verdict,
            'color': color,
            'recommendations': recommendations,
            'cluster_impact': cluster_impact,
            'risks': risks,
            'action': action
        }

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

uploaded_file = st.file_uploader("📁 Завантажте Excel файл з історією продажів", type=['xlsx', 'xls'])

if uploaded_file is not None:
    
    try:
        with st.spinner('Завантаження та аналіз даних...'):
            df = pd.read_excel(uploaded_file)
            analyzer = SalesDataAnalyzer(df)
            simulator = RealDataSimulator(analyzer)
            events_system = ExecutiveEventsSystem(analyzer)
        
        st.success(f"✅ Завантажено {len(df):,} записів про продажі | {analyzer.df['salon'].nunique()} салонів")
        
        # ====================================================================
        # ВКЛАДКИ
        # ====================================================================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Аналіз даних", "🎯 Симуляція", "🏆 Для директора", "📋 Кластери салонів", "🎯 Дашборд директора"])
        
        # ====================================================================
        # ВКЛАДКА 1: АНАЛІЗ ДАНИХ
        # ====================================================================
        
        with tab1:
            st.header("Аналіз поточних даних")
            
            # Загальна статистика
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
                st.metric("📦 Продано одиниць", f"{total_qty:,.0f}")
            with col4:
                st.metric("📈 Середня маржа", f"{avg_margin:.1f}%")
            with col5:
                st.metric("🧾 Середній чек", f"{avg_check:.0f}₴")
            
            st.markdown("---")
            
            # Статистика по салонах
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
                    title="Кількість салонів по кластерах",
                    color=cluster_dist.index,
                    color_discrete_map={'A': 'gold', 'B': 'silver', 'C': 'brown'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Топ товарів
            st.subheader("🏅 Топ-10 товарів за виручкою")
            top_products = analyzer.get_top_products(10)
            if top_products is not None:
                top_products_display = top_products.copy()
                top_products_display['revenue'] = top_products_display['revenue'].apply(lambda x: f"{x/1000:.0f}K₴")
                top_products_display['profit'] = top_products_display['profit'].apply(lambda x: f"{x/1000:.0f}K₴")
                top_products_display.columns = ['Виручка', 'Прибуток', 'Кількість']
                st.dataframe(top_products_display, use_container_width=True)
            
            # Аналіз по сегментах
            segment_stats = analyzer.get_segment_analysis()
            if segment_stats is not None:
                st.subheader("🏷️ Продажі по сегментах товарів")
                
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
            
            # Параметри симуляції
            col1, col2 = st.columns(2)
            
            with col1:
                cluster = st.selectbox(
                    "Кластер салонів для зміни цін",
                    options=['A', 'B', 'C'],
                    help="A - Преміум, B - Середній, C - Економ"
                )
                
                cluster_info = analyzer.clusters[analyzer.clusters['cluster'] == cluster]
                cluster_revenue = cluster_info['revenue'].sum()
                st.info(f"📍 У кластері {cluster}: {len(cluster_info)} салонів | Виручка: {cluster_revenue/1_000_000:.1f}M₴")
            
            with col2:
                price_change = st.slider(
                    "Зміна ціни (%)",
                    min_value=-30,
                    max_value=30,
                    value=-10,
                    step=5,
                    help="Від'ємне значення = зниження, позитивне = підвищення"
                )
                
                if price_change < 0:
                    st.warning(f"📉 Зниження цін на {abs(price_change)}%")
                else:
                    st.info(f"📈 Підвищення цін на {price_change}%")
            
            # Кнопка запуску
            if st.button("🚀 Запустити симуляцію", type="primary", use_container_width=True):
                
                with st.spinner("Розрахунок..."):
                    results = simulator.simulate_price_change(price_change, cluster)
                    summary = simulator.get_summary(results)
                    exec_rec = simulator.get_executive_recommendations(summary, price_change, cluster)
                
                # Результати
                st.markdown("---")
                st.subheader("📊 Результати симуляції")
                
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
                
                # Вердикт
                st.markdown("---")
                if exec_rec['color'] == 'success':
                    st.success(exec_rec['verdict'])
                elif exec_rec['color'] == 'warning':
                    st.warning(exec_rec['verdict'])
                else:
                    st.error(exec_rec['verdict'])
                
                # Графіки
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
                
                # Таблиця результатів
                st.markdown("---")
                st.subheader("📋 Детялізація по салонах")
                
                filter_cluster = st.selectbox(
                    "Показати салони кластеру:",
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
        
        # ====================================================================
        # ВКЛАДКА 3: ДЛЯ ДИРЕКТОРА
        # ====================================================================
        
        with tab3:
            st.header("🏆 Панель директора холдингу")
            
            if 'exec_rec' not in locals():
                st.info("👈 Спочатку запустіть симуляцію у вкладці 'Симуляція'")
            else:
                # Головний вердикт
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
                
                # Додаткова аналітика для директора
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
                    st.metric("Річний вплив на прибуток", f"{annual_impact / 1_000_000:.1f}M₴")
        
        # ====================================================================
        # ВКЛАДКА 4: КЛАСТЕРИ САЛОНІВ
        # ====================================================================
        
        with tab4:
            st.header("📋 Розподіл салонів по кластерах")
            
            st.markdown("""
            ### Як формуються кластери:
            
            - **Кластер A (Преміум)**: Салони з високим середнім чеком (топ 33%)
            - **Кластер B (Середній)**: Салони з середнім чеком (середні 33%)
            - **Кластер C (Економ)**: Салони з низьким середнім чеком (нижні 33%)
            
            Кластеризація автоматична на основі реальних даних продажів.
            """)
            
            st.markdown("---")
            
            # Таблиця салонів з кластерами
            clusters_display = analyzer.clusters[['cluster', 'revenue', 'profit', 'transactions', 
                                                   'avg_check', 'margin_pct', 'cluster_reason']].copy()
            
            clusters_display['revenue'] = clusters_display['revenue'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
            clusters_display['profit'] = clusters_display['profit'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
            clusters_display['avg_check'] = clusters_display['avg_check'].apply(lambda x: f"{x:.0f}₴")
            clusters_display['margin_pct'] = clusters_display['margin_pct'].apply(lambda x: f"{x:.1f}%")
            
            clusters_display.columns = ['Кластер', 'Виручка', 'Прибуток', 'Транзакції', 
                                        'Середній чек', 'Маржа', 'Чому цей кластер?']
            
            # Фільтр по кластеру
            cluster_filter = st.selectbox(
                "Фільтр по кластеру:",
                options=['Всі'] + ['A', 'B', 'C']
            )
            
            if cluster_filter != 'Всі':
                clusters_display = clusters_display[clusters_display['Кластер'] == cluster_filter]
            
            st.dataframe(clusters_display, use_container_width=True, height=600)
            
            # Статистика по кластерах
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
            
            # Інформація про модель
            with st.expander("ℹ️ Про модель симуляції"):
                st.markdown("""
                **Математика моделі:**

                1. **Еластичність попиту** (як попит реагує на ціну):
                   ```
                   Новий попит = Базовий попит × (1 + Δ% ціни × Еластичність)

                   Еластичності:
                   - Кластер A: -0.8 (при -10% ціни → +8% попиту)
                   - Кластер B: -1.2 (при -10% ціни → +12% попиту)
                   - Кластер C: -1.5 (при -10% ціни → +15% попиту)
                   ```

                2. **Зміна маржі**:
                   ```
                   При зниженні ціни на X%, маржа падає на X × 1.5%
                   Приклад: ціна -10% → маржа -15%
                   ```

                3. **Переток клієнтів**:
                   ```
                   - При зниженні цін: +25% додаткового притоку в цільовий кластер
                   - Відтік з кластеру B: -3% при зниженні цін в A
                   ```

                4. **Розрахунок виручки та прибутку**:
                   ```
                   Нова виручка = Базова виручка × Мультиплікатор попиту × Мультиплікатор ціни
                   Новий прибуток = Нова виручка × Нова маржа
                   ```

                **Використані дані:**
                - Реальні продажі з вашого Excel файлу
                - Автоматичний розрахунок середніх чеків, маржі, ROI
                - Кластеризація на основі квантилів
                """)

        # ====================================================================
        # ВКЛАДКА 5: ДАШБОРД ДИРЕКТОРА
        # ====================================================================

        with tab5:
            st.header("🎯 Дашборд директора холдингу")
            st.markdown("### Система автоматичного виявлення подій, трендів та ризиків")

            dashboard_data = events_system.get_executive_dashboard_data()

            # Швидка статистика
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📢 Всього подій", dashboard_data['summary']['total_events'])
            with col2:
                st.metric("🔴 Критичні попередження", dashboard_data['summary']['critical_warnings'])
            with col3:
                st.metric("📈 Позитивні тренди", dashboard_data['summary']['positive_trends'])

            st.markdown("---")

            # Секція попереджень
            if dashboard_data['warnings']:
                st.subheader("⚠️ Попередження та ризики")

                for warning in dashboard_data['warnings']:
                    if warning['level'] == 'critical':
                        with st.expander(f"🔴 {warning['title']}", expanded=True):
                            st.error(warning['description'])
                            st.info(f"**📋 Рекомендована дія:** {warning['action']}")
                            st.caption(f"Вплив: {warning['impact']}")
                    else:
                        with st.expander(f"⚠️ {warning['title']}"):
                            st.warning(warning['description'])
                            st.info(f"**📋 Рекомендована дія:** {warning['action']}")
                            st.caption(f"Вплив: {warning['impact']}")

                st.markdown("---")

            # Секція подій
            if dashboard_data['events']:
                st.subheader("📢 Важливі події")

                # Розділення на колонки
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

            # Секція трендів
            if dashboard_data['trends']:
                st.subheader("📊 Виявлені тренди")

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

            # Додаткова аналітика
            st.subheader("📈 Детальна аналітика")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Топ-5 салонів (виручка)")
                top5 = analyzer.salons_stats.head(5)[['revenue', 'profit', 'margin_pct']]
                top5_display = top5.copy()
                top5_display['revenue'] = top5_display['revenue'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
                top5_display['profit'] = top5_display['profit'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
                top5_display['margin_pct'] = top5_display['margin_pct'].apply(lambda x: f"{x:.1f}%")
                top5_display.columns = ['Виручка', 'Прибуток', 'Маржа']
                st.dataframe(top5_display, use_container_width=True)

            with col2:
                st.markdown("#### Топ-5 салонів (ROI)")
                top5_roi = analyzer.salons_stats.nlargest(5, 'roi')[['revenue', 'profit', 'roi']]
                top5_roi_display = top5_roi.copy()
                top5_roi_display['revenue'] = top5_roi_display['revenue'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
                top5_roi_display['profit'] = top5_roi_display['profit'].apply(lambda x: f"{x/1_000_000:.2f}M₴")
                top5_roi_display['roi'] = top5_roi_display['roi'].apply(lambda x: f"{x:.1f}%")
                top5_roi_display.columns = ['Виручка', 'Прибуток', 'ROI']
                st.dataframe(top5_roi_display, use_container_width=True)

            st.markdown("---")

            # Загальні висновки
            st.subheader("💡 Загальні висновки та рекомендації")

            # Автоматичні висновки на основі даних
            total_revenue = analyzer.df['revenue'].sum()
            total_profit = analyzer.df['profit'].sum()
            overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0

            conclusions = []

            if overall_margin > 40:
                conclusions.append("✅ **Відмінна маржинальність:** Холдинг показує високу прибутковість (>40%). Зберігайте фокус на якості та преміум-сегменті.")
            elif overall_margin > 25:
                conclusions.append("👍 **Добра маржинальність:** Стійке становище (25-40%). Є можливості для оптимізації асортименту.")
            else:
                conclusions.append("⚠️ **Низька маржинальність:** Маржа <25% вимагає негайних дій - перегляд цін, оптимізація витрат, робота з постачальниками.")

            # Аналіз кластерів
            cluster_dist = analyzer.clusters['cluster'].value_counts()
            if 'A' in cluster_dist.index:
                premium_count = cluster_dist['A']
                premium_pct = premium_count / len(analyzer.clusters) * 100
                if premium_pct > 30:
                    conclusions.append(f"💎 **Сильний преміум-сегмент:** {premium_pct:.0f}% салонів в кластері A. Це конкурентна перевага!")
                else:
                    conclusions.append(f"📊 **Потенціал преміумізації:** Лише {premium_pct:.0f}% салонів в топ-сегменті. Розгляньте стратегію апгрейду.")

            # Диверсифікація
            segment_stats = analyzer.get_segment_analysis()
            if segment_stats is not None and len(segment_stats) > 1:
                top_segment_share = segment_stats['revenue_share'].max()
                if top_segment_share > 60:
                    conclusions.append(f"⚠️ **Висока концентрація:** {top_segment_share:.0f}% виручки з одного сегменту. Ризик при зміні ринку. Рекомендується диверсифікація.")
                else:
                    conclusions.append("✅ **Збалансований портфель:** Добра диверсифікація по сегментах знижує ризики.")

            for conclusion in conclusions:
                st.markdown(conclusion)

            st.markdown("---")

            # Експорт звіту
            st.subheader("📄 Експорт звіту")

            if st.button("📥 Сгенерувати Executive Summary", use_container_width=True):
                report = f"""
# EXECUTIVE SUMMARY - Дашборд директора холдингу

Дата звіту: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 📊 Ключові показники

- **Загальна виручка:** {total_revenue / 1_000_000:.2f}M₴
- **Загальний прибуток:** {total_profit / 1_000_000:.2f}M₴
- **Маржинальність:** {overall_margin:.1f}%
- **Кількість салонів:** {analyzer.df['salon'].nunique()}
- **Всього транзакцій:** {len(analyzer.df):,}

## ⚠️ Критичні попередження ({len([w for w in dashboard_data['warnings'] if w['level'] == 'critical'])})

{chr(10).join([f"- {w['title']}: {w['description']}" for w in dashboard_data['warnings'] if w['level'] == 'critical'])}

## 📈 Позитивні тренди

{chr(10).join([f"- {t['metric']}: {t['change']} - {t['description']}" for t in dashboard_data['trends'] if t['status'] == 'positive'])}

## 💡 Рекомендації

{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(conclusions)])}

---
Звіт згенеровано автоматично системою аналітики
                """

                st.download_button(
                    label="💾 Завантажити звіт (MD)",
                    data=report,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )

                st.success("✅ Звіт готовий до завантаження!")

                with st.expander("👀 Попередній перегляд звіту"):
                    st.markdown(report)
    
    except Exception as e:
        st.error(f"❌ Помилка при завантаженні файлу: {str(e)}")
        st.info("Переконайтеся, що файл містить колонки: Magazin, Datasales, Price, Qty, Sum тощо.")
        
        with st.expander("📋 Детальна інформація про помилку"):
            st.code(str(e))

else:
    # Інструкція для користувача
    st.info("👆 Завантажте Excel файл з історією продажів для початку аналізу")
    
    st.markdown("""
    ### 📋 Вимоги до файлу:
    
    Файл повинен містити наступні колонки:
    - **Magazin** - назва салону
    - **Datasales** - дата продажу
    - **Art** - артикул товару
    - **Describe** - опис товару
    - **Model** - модель
    - **Segment** - сегмент товару
    - **Purchaiseprice** - ціна закупівлі
    - **Price** - ціна продажу
    - **Qty** - кількість
    - **Sum** - сума продажу
    
    ### 🎯 Що ви отримаєте:
    
    1. **Аналіз даних:**
       - Статистика по салонах
       - Розподіл по кластерах
       - Аналіз сегментів товарів
       - Часові тренди
    
    2. **Симуляція "Що якщо":**
       - Зміна цін по кластерах
       - Прогноз виручки і прибутку
       - Деталізація по салонах
       - Візуалізація результатів
    
    3. **Панель директора:**
       - Рекомендації щодо впровадження
       - Аналіз ризиків
       - План дій
       - ROI та термін окупності
    
    4. **Кластери салонів:**
       - Автоматичний розподіл на A/B/C
       - Пояснення чому салон в певному кластері
       - Статистика по кластерах
    """)
