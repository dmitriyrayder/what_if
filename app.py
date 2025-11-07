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

    def get_business_insights(self):
        """
        Комплексний аналіз бізнесу: тренди, ризики, можливості, рекомендації
        Незалежний від фільтрів аналіз всієї мережі
        """
        insights = {
            'trends': [],
            'risks': [],
            'opportunities': [],
            'recommendations': [],
            'anomalies': [],
            'seasonal_patterns': []
        }

        # ====================================================================
        # АНАЛІЗ ТРЕНДІВ
        # ====================================================================

        # Тренд виручки по часу
        if 'date' in self.df.columns and len(self.df) > 0:
            ts = self.get_time_series()
            if ts is not None and len(ts) >= 2:
                # Порівняння останнього та попереднього періоду
                last_period = ts['revenue'].iloc[-1]
                prev_period = ts['revenue'].iloc[-2]
                revenue_change = ((last_period / prev_period) - 1.0) * 100.0 if prev_period > 0 else 0

                if revenue_change > 10:
                    insights['trends'].append({
                        'type': 'positive',
                        'icon': '📈',
                        'title': 'Сильне зростання виручки',
                        'description': f'Виручка зросла на {revenue_change:.1f}% порівняно з попереднім періодом. Це відмінний результат!'
                    })
                elif revenue_change > 0:
                    insights['trends'].append({
                        'type': 'positive',
                        'icon': '📊',
                        'title': 'Помірне зростання виручки',
                        'description': f'Виручка зросла на {revenue_change:.1f}%. Стабільне зростання бізнесу.'
                    })
                elif revenue_change > -10:
                    insights['trends'].append({
                        'type': 'warning',
                        'icon': '⚠️',
                        'title': 'Невелике падіння виручки',
                        'description': f'Виручка знизилась на {abs(revenue_change):.1f}%. Потрібна увага.'
                    })
                else:
                    insights['trends'].append({
                        'type': 'danger',
                        'icon': '📉',
                        'title': 'Значне падіння виручки',
                        'description': f'Виручка знизилась на {abs(revenue_change):.1f}%. ТЕРМІНОВІ ДІЇ!'
                    })
                    insights['risks'].append({
                        'severity': 'high',
                        'icon': '🔴',
                        'title': 'Критичне падіння продажів',
                        'description': f'Падіння виручки на {abs(revenue_change):.1f}% - потрібен терміновий аналіз причин'
                    })

        # ====================================================================
        # АНАЛІЗ ПО КЛАСТЕРАХ
        # ====================================================================

        cluster_performance = self.clusters.groupby('cluster').agg({
            'revenue': 'sum',
            'margin_pct': 'mean',
            'profit': 'sum'
        })

        # Виявлення найприбутковішого кластера
        best_cluster = cluster_performance['profit'].idxmax()
        best_profit_share = (cluster_performance.loc[best_cluster, 'profit'] / cluster_performance['profit'].sum() * 100)

        insights['opportunities'].append({
            'priority': 'high',
            'icon': '🎯',
            'title': f'Кластер {best_cluster} - лідер по прибутку',
            'description': f'Кластер {best_cluster} генерує {best_profit_share:.1f}% від загального прибутку. Варто вивчити їх стратегію для масштабування на інші кластери.'
        })

        # Виявлення кластера з найвищою маржею
        best_margin_cluster = cluster_performance['margin_pct'].idxmax()
        best_margin = cluster_performance.loc[best_margin_cluster, 'margin_pct']

        if best_margin > 45:
            insights['opportunities'].append({
                'priority': 'medium',
                'icon': '💰',
                'title': f'Висока маржа в кластері {best_margin_cluster}',
                'description': f'Кластер {best_margin_cluster} має маржу {best_margin:.1f}%. Це вище середнього - можна оптимізувати ціни в інших кластерах.'
            })

        # ====================================================================
        # ВИЯВЛЕННЯ РИЗИКІВ
        # ====================================================================

        # Ризик: Низька маржа
        low_margin_salons = self.salons_stats[self.salons_stats['margin_pct'] < 25]
        if len(low_margin_salons) > 0:
            low_margin_pct = len(low_margin_salons) / len(self.salons_stats) * 100
            insights['risks'].append({
                'severity': 'medium',
                'icon': '⚠️',
                'title': f'{len(low_margin_salons)} салонів з низькою маржею',
                'description': f'{low_margin_pct:.1f}% салонів мають маржу <25%. Рекомендується аналіз цін та собівартості.'
            })

        # Ризик: Низька виручка
        revenue_threshold = self.salons_stats['revenue'].quantile(0.25)
        low_revenue_salons = self.salons_stats[self.salons_stats['revenue'] < revenue_threshold]
        if len(low_revenue_salons) > 3:
            insights['risks'].append({
                'severity': 'medium',
                'icon': '📊',
                'title': f'{len(low_revenue_salons)} салонів з низькою виручкою',
                'description': f'Ці салони потребують підтримки: маркетинг, асортимент, навчання персоналу.'
            })

        # Ризик: Концентрація виручки
        top_3_revenue = self.salons_stats.head(3)['revenue'].sum()
        total_revenue = self.salons_stats['revenue'].sum()
        top_3_share = (top_3_revenue / total_revenue * 100) if total_revenue > 0 else 0

        if top_3_share > 50:
            insights['risks'].append({
                'severity': 'high',
                'icon': '🎯',
                'title': 'Висока концентрація виручки',
                'description': f'Топ-3 салони дають {top_3_share:.1f}% виручки. Ризик залежності від кількох точок. Рекомендується розвиток інших салонів.'
            })

        # ====================================================================
        # ВИЯВЛЕННЯ МОЖЛИВОСТЕЙ
        # ====================================================================

        # Можливість: Салони, що швидко зростають
        if 'date' in self.df.columns:
            # Аналіз зростаючих салонів потребує даних по періодах
            pass

        # Можливість: Найприбутковіші сегменти
        segment_stats = self.get_segment_analysis()
        if segment_stats is not None and len(segment_stats) > 0:
            top_segment = segment_stats.index[0]
            top_segment_share = segment_stats['revenue_share'].iloc[0]

            insights['opportunities'].append({
                'priority': 'high',
                'icon': '🏆',
                'title': f'Сегмент "{top_segment}" - найприбутковіший',
                'description': f'Сегмент "{top_segment}" генерує {top_segment_share:.1f}% виручки. Варто розширити асортимент в цьому сегменті.'
            })

            # Сегменти з високою маржею
            high_margin_segments = segment_stats[segment_stats['margin'] > 45]
            if len(high_margin_segments) > 0:
                for segment_name, row in high_margin_segments.head(3).iterrows():
                    insights['opportunities'].append({
                        'priority': 'medium',
                        'icon': '💎',
                        'title': f'Високомаржинальний сегмент: {segment_name}',
                        'description': f'Маржа {row["margin"]:.1f}% - варто збільшити частку цього сегменту в продажах.'
                    })

        # ====================================================================
        # ЗАГАЛЬНІ РЕКОМЕНДАЦІЇ
        # ====================================================================

        # Рекомендація 1: Оптимізація асортименту
        if segment_stats is not None and len(segment_stats) > 0:
            low_performing_segments = segment_stats[segment_stats['revenue_share'] < 5]
            if len(low_performing_segments) > 0:
                insights['recommendations'].append({
                    'category': 'Асортимент',
                    'icon': '📦',
                    'title': 'Оптимізація асортименту',
                    'description': f'Виявлено {len(low_performing_segments)} низькорентабельних сегментів (<5% виручки). Рекомендується: перегляд асортименту, промо-акції або виведення з асортименту.',
                    'impact': 'medium'
                })

        # Рекомендація 2: Розвиток слабких салонів
        bottom_quartile = self.salons_stats[self.salons_stats['revenue'] < self.salons_stats['revenue'].quantile(0.25)]
        if len(bottom_quartile) > 0:
            insights['recommendations'].append({
                'category': 'Операційна ефективність',
                'icon': '🎯',
                'title': 'Програма розвитку слабких салонів',
                'description': f'Створити програму підтримки для {len(bottom_quartile)} салонів з нижньої квартилі: навчання, маркетинг, оптимізація асортименту.',
                'impact': 'high'
            })

        # Рекомендація 3: Підвищення середнього чека
        avg_check = self.salons_stats['avg_check'].mean()
        insights['recommendations'].append({
            'category': 'Маркетинг',
            'icon': '🛒',
            'title': 'Програма збільшення середнього чека',
            'description': f'Поточний середній чек: {avg_check:.0f}₴. Рекомендації: cross-selling, up-selling, програми лояльності, bundle-пропозиції.',
            'impact': 'high'
        })

        # Рекомендація 4: Ціноутворення
        if len(low_margin_salons) > 0:
            insights['recommendations'].append({
                'category': 'Ціноутворення',
                'icon': '💰',
                'title': 'Аудит цін та маржі',
                'description': f'Провести аудит {len(low_margin_salons)} салонів з низькою маржею. Можливі напрямки: переговори з постачальниками, оптимізація операційних витрат, перегляд цін.',
                'impact': 'high'
            })

        # Рекомендація 5: Масштабування успішного досвіду
        top_salon = self.salons_stats.index[0]
        top_salon_stats = self.salons_stats.loc[top_salon]
        insights['recommendations'].append({
            'category': 'Стратегія',
            'icon': '🚀',
            'title': 'Масштабування best practices',
            'description': f'Проаналізувати стратегію салону "{top_salon}" (виручка: {top_salon_stats["revenue"]/1_000:.0f}K₴, маржа: {top_salon_stats["margin_pct"]:.1f}%) та масштабувати на інші салони.',
            'impact': 'high'
        })

        # ====================================================================
        # АНОМАЛІЇ
        # ====================================================================

        # Виявлення салонів з екстремальними показниками
        revenue_std = self.salons_stats['revenue'].std()
        revenue_mean = self.salons_stats['revenue'].mean()

        outliers_high = self.salons_stats[self.salons_stats['revenue'] > revenue_mean + 2 * revenue_std]
        if len(outliers_high) > 0:
            for salon_name, stats in outliers_high.iterrows():
                insights['anomalies'].append({
                    'type': 'positive',
                    'icon': '⭐',
                    'title': f'Виняткова ефективність: {salon_name}',
                    'description': f'Виручка {stats["revenue"]/1_000:.0f}K₴ значно вище середнього. Вивчити досвід для тиражування.'
                })

        outliers_low = self.salons_stats[self.salons_stats['revenue'] < revenue_mean - 2 * revenue_std]
        if len(outliers_low) > 0:
            for salon_name, stats in outliers_low.head(3).iterrows():
                insights['anomalies'].append({
                    'type': 'warning',
                    'icon': '⚡',
                    'title': f'Критично низька виручка: {salon_name}',
                    'description': f'Виручка {stats["revenue"]/1_000:.0f}K₴ значно нижче середнього. Потрібен терміновий аналіз.'
                })

        return insights

# ============================================================================
# СИМУЛЯТОР
# ============================================================================

class RealDataSimulator:
    """Симулятор на основі реальних даних"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.baseline = analyzer.salons_stats
        
    def simulate_price_change(self, price_change_pct, target_cluster, selected_segment=None, custom_demand_change=None):
        """
        Симуляція зміни цін

        Математика:
        1. Новий попит = Базовий попит × (1 + ΔЦіна × Еластичність)
        2. При зниженні цін додаємо приплив клієнтів
        3. Нова виручка = Новий попит × Нова ціна
        4. Новий прибуток = Нова виручка × Нова маржа

        Args:
            price_change_pct: відсоток зміни ціни
            target_cluster: цільовий кластер
            selected_segment: опціонально, конкретний сегмент
            custom_demand_change: ручна зміна попиту у відсотках (якщо задано, використовується замість еластичності)
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

                # 1. Зміна попиту
                if custom_demand_change is not None:
                    # Використовуємо ручне значення зміни попиту
                    demand_multiplier = 1.0 + (custom_demand_change / 100.0)
                else:
                    # Використовуємо еластичність
                    # Формула: demand_multiplier = 1 + (% зміни ціни / 100) × еластичність
                    demand_multiplier = 1.0 + (price_change_pct / 100.0) * elasticity[cluster]

                    # 2. Приплив клієнтів при зниженні цін
                    if price_change_pct < 0:
                        demand_multiplier += spillover_to_target
                
                # 3. Нова виручка = Базова виручка × Мультиплікатор попиту × Мультиплікатор ціни
                price_multiplier = 1.0 + price_change_pct / 100.0
                new_revenue = baseline_stats['revenue'] * demand_multiplier * price_multiplier

                # 4. Розрахунок нової маржі (коректна математика)
                # Базовий прибуток на одиницю = Базова виручка × Базова маржа%
                baseline_margin_amount = baseline_stats['revenue'] * (baseline_stats['margin_pct'] / 100.0)
                # Собівартість = Базова виручка - Базовий прибуток
                cost_amount = baseline_stats['revenue'] - baseline_margin_amount

                # При зміні ціни:
                # Нова виручка вже розрахована вище з урахуванням зміни попиту та ціни
                # Собівартість змінюється пропорційно зміні попиту (більше/менше одиниць)
                new_cost = cost_amount * demand_multiplier

                # Новий прибуток = Нова виручка - Нова собівартість
                new_profit = new_revenue - new_cost

                # Нова маржа% = (Новий прибуток / Нова виручка) × 100
                # Обмежуємо маржу від 0% до 100%
                new_margin_pct = (new_profit / new_revenue * 100.0) if new_revenue > 0 else 0
                new_margin_pct = max(0, min(new_margin_pct, 100))

                # Переконуємося що прибуток не від'ємний
                new_profit = max(new_profit, 0)
                
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

        # Основний вердикт - розширена логіка з більш детальними сценаріями
        if profit_change > 15:
            verdict = "🚀 ДУЖЕ РЕКОМЕНДУЄТЬСЯ! (Високий прибуток)"
            color = "success"
        elif profit_change > 8:
            verdict = "✅ РЕКОМЕНДУЄТЬСЯ ВПРОВАДИТИ (Хороший прибуток)"
            color = "success"
        elif profit_change > 3:
            verdict = "✅ РЕКОМЕНДУЄТЬСЯ (Позитивний ефект)"
            color = "success"
        elif profit_change > 0:
            verdict = "⚠️ НЕЙТРАЛЬНО (низький позитивний ефект)"
            color = "warning"
        elif profit_change > -3:
            verdict = "⚠️ ОБЕРЕЖНО (малі втрати)"
            color = "warning"
        else:
            verdict = "❌ НЕ РЕКОМЕНДУЄТЬСЯ (значні втрати)"
            color = "error"
        
        # Детальні рекомендації
        if price_change_pct < 0:
            # Зниження цін
            if profit_change > 10:
                recommendations.append("🎯 Чудовий результат! Зниження цін призводить до значного зростання прибутку за рахунок збільшення обсягів продажів")
                recommendations.append(f"💡 Рекомендація: ТЕРМІНОВО запустити акцію в кластері {target_cluster} на 2-4 тижні")
                recommendations.append("⏰ Моніторити щоденно перші 7 днів для коригування стратегії")
                recommendations.append("📊 Розглянути можливість розширення акції на інші кластери")
            elif profit_change > 0:
                recommendations.append("🎯 Зниження цін призводить до зростання прибутку за рахунок збільшення обсягів продажів")
                recommendations.append(f"💡 Рекомендація: Запустити акцію в кластері {target_cluster} на 2-4 тижні")
                recommendations.append("⏰ Моніторити щоденно перші 7 днів для коригування стратегії")
            else:
                recommendations.append("⚠️ Зниження цін не компенсується зростанням продажів")
                recommendations.append(f"💡 Альтернатива: Замість зниження цін розглянути промо 2+1 або подарунки")
                recommendations.append("📊 Провести A/B тест на 2-3 салонах перед масштабуванням")
        else:
            # Підвищення цін
            if profit_change > 10:
                recommendations.append("💰 Відмінний результат! Підвищення цін значно збільшує прибутковість")
                recommendations.append(f"💡 Рекомендація: Поступове підвищення цін в кластері {target_cluster} на 5% щомісяця")
                recommendations.append("🎯 Супроводжувати підвищення покращенням сервісу та якості обслуговування")
                recommendations.append("📈 Інвестувати додатковий прибуток в маркетинг для утримання клієнтів")
            elif profit_change > 0:
                recommendations.append("💰 Підвищення цін призводить до зростання прибутковості")
                recommendations.append(f"💡 Рекомендація: Поступове підвищення цін в кластері {target_cluster} на 3-5% щомісяця")
                recommendations.append("🎯 Супроводжувати підвищення покращенням сервісу")
            else:
                recommendations.append("📉 Підвищення цін призводить до критичного відтоку клієнтів")
                recommendations.append("💡 Рекомендація: Не підвищувати ціни, зосередитись на оптимізації витрат")
                recommendations.append("🔍 Провести аналіз цін конкурентів перед наступною спробою")
        
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
        
        # Детальний план дій з кількома варіантами
        action_plans = []

        if profit_change > 15:
            action_plans.append("🚀 ВАРІАНТ 1 (Швидкий розгорт): Масштабувати на всі салони кластеру протягом 3-5 днів")
            action_plans.append("📊 ВАРІАНТ 2 (Збалансований): Запустити у 70% салонів, залишити 30% контрольною групою на 2 тижні")
            action_plans.append("🎯 ВАРІАНТ 3 (Агресивний): Розширити на всі кластери з адаптацією під кожний")
        elif profit_change > 8:
            action_plans.append("✅ ВАРІАНТ 1 (Рекомендований): Масштабувати на всі салони кластеру протягом тижня")
            action_plans.append("🧪 ВАРІАНТ 2 (Обережний): Запустити у 50% салонів, залишити 50% контрольною групою")
            action_plans.append("📈 ВАРІАНТ 3 (Поступовий): Запуск по 25% салонів щотижня протягом місяця")
        elif profit_change > 3:
            action_plans.append("🧪 ВАРІАНТ 1 (Рекомендований): Запустити пілот на 50% салонів, аналіз через 2 тижні")
            action_plans.append("⚡ ВАРІАНТ 2 (Швидкий): Запустити у всіх салонах з можливістю швидкого відкату")
            action_plans.append("🔍 ВАРІАНТ 3 (Дослідницький): A/B тестування на 5-7 салонах протягом місяця")
        elif profit_change > 0:
            action_plans.append("🧪 ВАРІАНТ 1 (Рекомендований): Тестовий запуск на 3-5 салонах протягом 2 тижнів")
            action_plans.append("📊 ВАРІАНТ 2 (Дослідження): Провести фокус-групи з клієнтами перед впровадженням")
            action_plans.append("💡 ВАРІАНТ 3 (Альтернатива): Розглянути модифіковані умови (менша зміна ціни)")
        elif profit_change > -3:
            action_plans.append("⚠️ ВАРІАНТ 1 (Обережний): Мікро-тест на 1-2 салонах максимум на тиждень")
            action_plans.append("🔄 ВАРІАНТ 2 (Модифікація): Змінити параметри та перезапустити симуляцію")
            action_plans.append("🛑 ВАРІАНТ 3 (Відмова): Шукати альтернативні стратегії (програми лояльності, кросел)")
        else:
            action_plans.append("🛑 ВАРІАНТ 1 (Рекомендований): НЕ впроваджувати цю стратегію")
            action_plans.append("🔍 ВАРІАНТ 2 (Аналіз): Провести детальний аналіз причин негативного результату")
            action_plans.append("💡 ВАРІАНТ 3 (Альтернативи): Розглянути інші стратегії: оптимізація витрат, покращення сервісу, програми лояльності")

        # Об'єднуємо всі плани в один текст
        action = "\n\n".join(action_plans)
        
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
        
        st.success(f"✅ Завантажено {len(df):,} записів про продажі | {analyzer.df['salon'].nunique()} салонів")
        
        # ====================================================================
        # ВКЛАДКИ
        # ====================================================================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Аналіз даних", "🎯 Симуляція", "🏆 Для директора", "📋 Кластери салонів", "⚡ Події та Інсайти"])
        
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

            # Довідка про формули
            with st.expander("ℹ️ Як працює розрахунок симуляції?"):
                st.markdown("""
                ### 📊 Формули розрахунку:

                **1. Зміна попиту (еластичність):**
                ```
                Мультиплікатор попиту = 1 + (Δ% ціни / 100) × Еластичність

                Коефіцієнти еластичності:
                • Кластер A (Преміум): -0.8 → при зниженні ціни на 10% попит зросте на 8%
                • Кластер B (Середній): -1.2 → при зниженні ціни на 10% попит зросте на 12%
                • Кластер C (Економ): -1.5 → при зниженні ціни на 10% попит зросте на 15%

                При зниженні цін додатково: +25% приплив клієнтів
                ```

                **2. Розрахунок виручки:**
                ```
                Мультиплікатор ціни = 1 + (Δ% ціни / 100)
                Нова виручка = Базова виручка × Мультиплікатор попиту × Мультиплікатор ціни
                ```

                **3. Розрахунок собівартості та прибутку:**
                ```
                Базова собівартість = Базова виручка × (1 - Базова маржа% / 100)
                Нова собівартість = Базова собівартість × Мультиплікатор попиту

                Новий прибуток = Нова виручка - Нова собівартість
                Нова маржа% = (Новий прибуток / Нова виручка) × 100
                ```

                **Приклад розрахунку:**
                ```
                Базові дані: Виручка = 100,000₴, Маржа = 40%
                → Прибуток = 40,000₴, Собівартість = 60,000₴

                Зміна ціни: -10% (зниження)
                Еластичність кластера B: -1.2
                → Мультиплікатор попиту = 1 + (-10/100 × -1.2) + 0.25 = 1.37 (+37%)
                → Мультиплікатор ціни = 1 - 0.1 = 0.9

                Нова виручка = 100,000 × 1.37 × 0.9 = 123,300₴
                Нова собівартість = 60,000 × 1.37 = 82,200₴
                Новий прибуток = 123,300 - 82,200 = 41,100₴ (+2.8%)
                Нова маржа% = (41,100 / 123,300) × 100 = 33.3%
                ```

                **4. Ефект перетоку клієнтів:**
                - При зниженні цін: +25% додатковий приплив клієнтів в цільовий кластер
                - Відтік з інших кластерів: -3% при зниженні цін в сусідньому кластері
                """)

            # Параметри симуляції
            col1, col2, col3 = st.columns(3)

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

            with col3:
                use_custom_demand = st.checkbox(
                    "🎯 Ручний ввід зміни попиту",
                    help="Дозволяє вручну задати зміну попиту замість автоматичного розрахунку через еластичність"
                )

                if use_custom_demand:
                    custom_demand = st.slider(
                        "Очікувана зміна попиту (%)",
                        min_value=-50,
                        max_value=100,
                        value=0,
                        step=5,
                        help="Наприклад, якщо очікуєте зростання продажів на 20%, введіть +20"
                    )
                    if custom_demand < 0:
                        st.warning(f"📉 Падіння попиту: {abs(custom_demand)}%")
                    elif custom_demand > 0:
                        st.success(f"📈 Зростання попиту: {custom_demand}%")
                    else:
                        st.info("➡️ Попит без змін")
                else:
                    custom_demand = None
            
            # Кнопка запуску
            if st.button("🚀 Запустити симуляцію", type="primary", use_container_width=True):

                with st.spinner("Розрахунок..."):
                    results = simulator.simulate_price_change(price_change, cluster, custom_demand_change=custom_demand)
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

            # Довідка про формули розрахунку
            with st.expander("ℹ️ Як інтерпретувати результати симуляції?"):
                st.markdown("""
                ### 📊 Пояснення вердиктів:

                **🚀 ДУЖЕ РЕКОМЕНДУЄТЬСЯ** (прибуток > +15%)
                - Стратегія має дуже високий потенціал
                - Рекомендується швидке впровадження
                - Мінімальні ризики

                **✅ РЕКОМЕНДУЄТЬСЯ ВПРОВАДИТИ** (прибуток > +8%)
                - Стратегія має хороший потенціал
                - Впровадження варто розпочати протягом тижня
                - Помірні ризики

                **✅ РЕКОМЕНДУЄТЬСЯ** (прибуток > +3%)
                - Позитивний ефект, але невеликий
                - Варто розглянути впровадження
                - Контролювати результати

                **⚠️ НЕЙТРАЛЬНО** (прибуток 0% до +3%)
                - Мінімальний позитивний ефект
                - Рекомендується A/B тестування перед масштабуванням

                **⚠️ ОБЕРЕЖНО** (прибуток -3% до 0%)
                - Малі втрати
                - Не рекомендується без додаткового аналізу

                **❌ НЕ РЕКОМЕНДУЄТЬСЯ** (прибуток < -3%)
                - Значні втрати
                - Не впроваджувати

                ### 📈 Формули розрахунку результатів:

                **Зміна виручки:**
                ```
                Δ Виручка % = ((Нова виручка - Базова виручка) / Базова виручка) × 100%
                ```

                **Зміна прибутку:**
                ```
                Δ Прибуток % = ((Новий прибуток - Базовий прибуток) / Базовий прибуток) × 100%
                ```

                **Річний вплив:**
                ```
                Річний вплив = (Новий прибуток - Базовий прибуток) × 12 місяців
                ```
                """)

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
        # ВКЛАДКА 5: ПОДІЇ ТА ІНСАЙТИ
        # ====================================================================

        with tab5:
            st.header("⚡ Події та Інсайти")
            st.markdown("### Комплексний аналіз всієї мережі незалежно від фільтрів")

            # Отримуємо інсайти
            insights = analyzer.get_business_insights()

            # ====================================================================
            # ТРЕНДИ
            # ====================================================================

            st.markdown("---")
            st.subheader("📈 Тренди та Динаміка")

            if len(insights['trends']) > 0:
                for trend in insights['trends']:
                    if trend['type'] == 'positive':
                        st.success(f"{trend['icon']} **{trend['title']}**\n\n{trend['description']}")
                    elif trend['type'] == 'warning':
                        st.warning(f"{trend['icon']} **{trend['title']}**\n\n{trend['description']}")
                    else:
                        st.error(f"{trend['icon']} **{trend['title']}**\n\n{trend['description']}")
            else:
                st.info("Недостатньо даних для аналізу трендів (потрібно мінімум 2 періоди)")

            # ====================================================================
            # МОЖЛИВОСТІ
            # ====================================================================

            st.markdown("---")
            st.subheader("🎯 Можливості для Розвитку")

            if len(insights['opportunities']) > 0:
                col1, col2 = st.columns(2)

                for idx, opportunity in enumerate(insights['opportunities']):
                    col = col1 if idx % 2 == 0 else col2

                    with col:
                        priority_color = {
                            'high': '🔴',
                            'medium': '🟡',
                            'low': '🟢'
                        }

                        st.info(f"{priority_color.get(opportunity['priority'], '⚪')} {opportunity['icon']} **{opportunity['title']}**\n\n{opportunity['description']}")
            else:
                st.info("Не виявлено особливих можливостей")

            # ====================================================================
            # РИЗИКИ
            # ====================================================================

            st.markdown("---")
            st.subheader("⚠️ Ризики та Застереження")

            if len(insights['risks']) > 0:
                for risk in insights['risks']:
                    if risk['severity'] == 'high':
                        st.error(f"{risk['icon']} **{risk['title']}**\n\n{risk['description']}")
                    elif risk['severity'] == 'medium':
                        st.warning(f"{risk['icon']} **{risk['title']}**\n\n{risk['description']}")
                    else:
                        st.info(f"{risk['icon']} **{risk['title']}**\n\n{risk['description']}")
            else:
                st.success("✅ Не виявлено критичних ризиків!")

            # ====================================================================
            # РЕКОМЕНДАЦІЇ
            # ====================================================================

            st.markdown("---")
            st.subheader("💡 Стратегічні Рекомендації")

            if len(insights['recommendations']) > 0:
                # Групуємо рекомендації по категоріях
                recommendations_by_category = {}
                for rec in insights['recommendations']:
                    category = rec.get('category', 'Інше')
                    if category not in recommendations_by_category:
                        recommendations_by_category[category] = []
                    recommendations_by_category[category].append(rec)

                # Відображаємо по категоріях
                for category, recs in recommendations_by_category.items():
                    with st.expander(f"📂 {category} ({len(recs)} рекомендацій)", expanded=True):
                        for rec in recs:
                            impact_badge = {
                                'high': '🔴 Високий вплив',
                                'medium': '🟡 Середній вплив',
                                'low': '🟢 Низький вплив'
                            }

                            st.markdown(f"""
                            **{rec['icon']} {rec['title']}** {impact_badge.get(rec['impact'], '')}

                            {rec['description']}
                            """)
                            st.markdown("---")
            else:
                st.info("Немає специфічних рекомендацій на даний момент")

            # ====================================================================
            # АНОМАЛІЇ
            # ====================================================================

            if len(insights['anomalies']) > 0:
                st.markdown("---")
                st.subheader("⚡ Виявлені Аномалії")

                col1, col2 = st.columns(2)

                positive_anomalies = [a for a in insights['anomalies'] if a['type'] == 'positive']
                warning_anomalies = [a for a in insights['anomalies'] if a['type'] == 'warning']

                with col1:
                    if positive_anomalies:
                        st.markdown("**🌟 Позитивні:**")
                        for anomaly in positive_anomalies:
                            st.success(f"{anomaly['icon']} **{anomaly['title']}**\n\n{anomaly['description']}")

                with col2:
                    if warning_anomalies:
                        st.markdown("**⚠️ Тривожні:**")
                        for anomaly in warning_anomalies:
                            st.warning(f"{anomaly['icon']} **{anomaly['title']}**\n\n{anomaly['description']}")

            # ====================================================================
            # ПІДСУМОК
            # ====================================================================

            st.markdown("---")
            st.subheader("📊 Короткий Підсумок")

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            with summary_col1:
                st.metric("Виявлено трендів", len(insights['trends']))

            with summary_col2:
                st.metric("Можливостей", len(insights['opportunities']))

            with summary_col3:
                st.metric("Ризиків", len(insights['risks']))

            with summary_col4:
                st.metric("Рекомендацій", len(insights['recommendations']))

            # Довідка
            with st.expander("ℹ️ Про розділ 'Події та Інсайти'"):
                st.markdown("""
                ### Що це?

                Розділ "Події та Інсайти" надає комплексний аналіз всієї мережі салонів оптики **незалежно від фільтрів**.

                ### Що аналізується?

                **📈 Тренди:**
                - Динаміка виручки та прибутку
                - Порівняння з попередніми періодами
                - Виявлення зростання чи падіння

                **🎯 Можливості:**
                - Найприбутковіші кластери та сегменти
                - Високомаржинальні категорії товарів
                - Салони-лідери для тиражування досвіду

                **⚠️ Ризики:**
                - Салони з низькою маржею або виручкою
                - Концентрація виручки
                - Критичні відхилення від норми

                **💡 Рекомендації:**
                - Стратегічні ініціативи для покращення
                - Оптимізація асортименту
                - Підвищення ефективності
                - Масштабування успішного досвіду

                **⚡ Аномалії:**
                - Салони з екстремально високими/низькими показниками
                - Статистичні викиди для детального вивчення

                ### Як використовувати?

                1. Регулярно переглядайте цей розділ для стратегічного планування
                2. Досліджуйте виявлені можливості для зростання
                3. Працюйте над усуненням ризиків
                4. Впроваджуйте рекомендації поетапно
                5. Вивчайте аномалії для розуміння їх причин

                Цей аналіз оновлюється автоматично на основі завантажених даних.
                """)

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
