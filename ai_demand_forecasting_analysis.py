#!/usr/bin/env python3
"""
AI-Driven Demand Forecasting and Inventory Management Analysis
=============================================================

This script creates comprehensive visualizations and analysis for AI-driven
demand prediction, including time-series forecasting, clustering analysis,
and sales data distribution patterns.

Author: Research Analysis
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set academic plotting style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2

class AIDemandforcestingAnalysis:
    """
    Creates comprehensive analysis and visualizations for AI-driven demand forecasting.
    """
    
    def __init__(self, random_state=42):
        """Initialize the analysis with consistent random state."""
        self.random_state = random_state
        np.random.seed(random_state)
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#52B788',
            'warning': '#F77F00',
            'danger': '#C73E1D',
            'info': '#4ECDC4'
        }
        
    def generate_sales_data(self, years=3, products=50):
        """
        Generate synthetic sales data with seasonal patterns, trends, and noise.
        """
        # Create date range
        start_date = '2021-01-01'
        end_date = '2023-12-31'
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Product categories with different characteristics
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        product_data = []
        
        for i in range(products):
            category = np.random.choice(categories)
            product_name = f"Product_{category}_{i+1}"
            
            # Base demand level (varies by category)
            base_demand = {
                'Electronics': np.random.uniform(50, 200),
                'Clothing': np.random.uniform(30, 150),
                'Home & Garden': np.random.uniform(20, 100),
                'Sports': np.random.uniform(25, 120),
                'Books': np.random.uniform(10, 80)
            }[category]
            
            # Seasonal patterns (varies by category)
            seasonal_amplitude = {
                'Electronics': 0.3,  # Higher during holiday seasons
                'Clothing': 0.4,     # Strong seasonal variation
                'Home & Garden': 0.25,
                'Sports': 0.35,
                'Books': 0.15
            }[category]
            
            # Generate daily sales with multiple components
            daily_sales = []
            for j, date in enumerate(date_range):
                # Trend component (slight growth over time)
                trend = base_demand * (1 + 0.001 * j)
                
                # Seasonal component (yearly cycle)
                day_of_year = date.dayofyear
                seasonal = seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365.25)
                
                # Weekly pattern (higher sales on weekends for some categories)
                weekly_factor = 1.0
                if category in ['Electronics', 'Clothing'] and date.weekday() >= 5:
                    weekly_factor = 1.2
                
                # Monthly pattern (higher sales at month end/beginning)
                if date.day <= 5 or date.day >= 25:
                    monthly_factor = 1.1
                else:
                    monthly_factor = 1.0
                
                # Special events (Black Friday, Christmas, etc.)
                special_factor = 1.0
                if date.month == 11 and date.day >= 20:  # Black Friday period
                    special_factor = 1.5
                elif date.month == 12 and date.day >= 15:  # Christmas period
                    special_factor = 1.3
                elif date.month == 1 and date.day <= 15:  # New Year sales
                    special_factor = 0.8
                
                # Random noise
                noise = np.random.normal(0, 0.1 * base_demand)
                
                # Combine all components
                total_demand = max(0, trend * (1 + seasonal) * weekly_factor * monthly_factor * special_factor + noise)
                daily_sales.append({
                    'Date': date,
                    'Product': product_name,
                    'Category': category,
                    'Sales': total_demand,
                    'Year': date.year,
                    'Quarter': f"Q{(date.month-1)//3 + 1}",
                    'Month': date.strftime('%B'),
                    'Month_Num': date.month,
                    'Day_of_Week': date.strftime('%A'),
                    'Day_of_Year': day_of_year,
                    'Trend': trend,
                    'Seasonal': seasonal,
                    'Base_Demand': base_demand
                })
            
            product_data.extend(daily_sales)
        
        self.sales_df = pd.DataFrame(product_data)
        return self.sales_df
    
    def create_sales_distribution_plots(self):
        """
        Create comprehensive sales distribution plots by Year, Quarter, and Month.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Aggregate data for different time periods
        yearly_sales = self.sales_df.groupby(['Year', 'Category'])['Sales'].sum().reset_index()
        quarterly_sales = self.sales_df.groupby(['Year', 'Quarter', 'Category'])['Sales'].sum().reset_index()
        monthly_sales = self.sales_df.groupby(['Month_Num', 'Category'])['Sales'].sum().reset_index()
        
        # Plot 1: Sales by Year
        ax1 = axes[0, 0]
        pivot_yearly = yearly_sales.pivot(index='Year', columns='Category', values='Sales')
        pivot_yearly.plot(kind='bar', ax=ax1, color=[self.colors['primary'], self.colors['secondary'], 
                                                   self.colors['accent'], self.colors['success'], 
                                                   self.colors['warning']])
        ax1.set_title('Total Sales Distribution by Year', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Total Sales (Units)')
        ax1.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=0)
        
        # Plot 2: Sales by Quarter (stacked)
        ax2 = axes[0, 1]
        quarterly_pivot = quarterly_sales.groupby(['Quarter', 'Category'])['Sales'].mean().reset_index()
        quarterly_pivot = quarterly_pivot.pivot(index='Quarter', columns='Category', values='Sales')
        quarterly_pivot.plot(kind='bar', stacked=True, ax=ax2, 
                           color=[self.colors['primary'], self.colors['secondary'], 
                                 self.colors['accent'], self.colors['success'], 
                                 self.colors['warning']])
        ax2.set_title('Average Sales Distribution by Quarter', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Quarter')
        ax2.set_ylabel('Average Sales (Units)')
        ax2.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=0)
        
        # Plot 3: Sales by Month (line plot)
        ax3 = axes[0, 2]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pivot = monthly_sales.pivot(index='Month_Num', columns='Category', values='Sales')
        for i, category in enumerate(monthly_pivot.columns):
            ax3.plot(monthly_pivot.index, monthly_pivot[category], 
                    marker='o', linewidth=2, label=category, 
                    color=[self.colors['primary'], self.colors['secondary'], 
                          self.colors['accent'], self.colors['success'], 
                          self.colors['warning']][i])
        ax3.set_title('Seasonal Sales Patterns by Month', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Total Sales (Units)')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(month_names)
        ax3.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time series for top product
        ax4 = axes[1, 0]
        top_product = self.sales_df.groupby('Product')['Sales'].sum().idxmax()
        product_ts = self.sales_df[self.sales_df['Product'] == top_product].set_index('Date')['Sales']
        ax4.plot(product_ts.index, product_ts.values, color=self.colors['primary'], linewidth=1.5)
        ax4.set_title(f'Time Series: {top_product}', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Daily Sales (Units)')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Sales distribution by day of week
        ax5 = axes[1, 1]
        dow_sales = self.sales_df.groupby('Day_of_Week')['Sales'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        bars = ax5.bar(range(len(dow_sales)), dow_sales.values, 
                      color=[self.colors['info'] if day in ['Saturday', 'Sunday'] 
                            else self.colors['primary'] for day in dow_sales.index])
        ax5.set_title('Average Sales by Day of Week', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Day of Week')
        ax5.set_ylabel('Average Sales (Units)')
        ax5.set_xticks(range(len(dow_sales)))
        ax5.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Plot 6: Category performance comparison
        ax6 = axes[1, 2]
        category_stats = self.sales_df.groupby('Category')['Sales'].agg(['mean', 'std']).reset_index()
        x = np.arange(len(category_stats))
        bars = ax6.bar(x, category_stats['mean'], yerr=category_stats['std'], 
                      capsize=5, color=[self.colors['primary'], self.colors['secondary'], 
                                       self.colors['accent'], self.colors['success'], 
                                       self.colors['warning']])
        ax6.set_title('Category Performance (Mean ± Std)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Category')
        ax6.set_ylabel('Sales (Units)')
        ax6.set_xticks(x)
        ax6.set_xticklabels(category_stats['Category'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('/Users/mayankdw/fast_delivery_paper/sales_distribution_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_demand_forecasting_visualization(self):
        """
        Create demand forecasting visualization with multiple algorithms.
        """
        # Select a representative product for forecasting demo
        product_sample = self.sales_df[self.sales_df['Product'] == 
                                     self.sales_df.groupby('Product')['Sales'].sum().idxmax()]
        
        # Prepare time series data
        ts_data = product_sample.set_index('Date')['Sales'].resample('D').sum()
        
        # Split into train and test
        split_point = int(len(ts_data) * 0.8)
        train_data = ts_data[:split_point]
        test_data = ts_data[split_point:]
        
        # Simple forecasting methods for demonstration
        # Moving average
        window_size = 30
        ma_forecast = train_data.rolling(window=window_size).mean().iloc[-1]
        ma_pred = [ma_forecast] * len(test_data)
        
        # Linear trend
        x_train = np.arange(len(train_data))
        y_train = train_data.values
        trend_coef = np.polyfit(x_train, y_train, 1)
        
        x_test = np.arange(len(train_data), len(train_data) + len(test_data))
        trend_pred = np.polyval(trend_coef, x_test)
        
        # Seasonal naive (use same day from previous year)
        seasonal_pred = []
        for i in range(len(test_data)):
            if i < len(train_data):
                seasonal_pred.append(train_data.iloc[-(365-i) if 365-i < len(train_data) else -1])
            else:
                seasonal_pred.append(train_data.iloc[-1])
        
        # Create forecasting visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Time series with forecasts
        ax1 = axes[0, 0]
        ax1.plot(train_data.index, train_data.values, label='Training Data', 
                color=self.colors['primary'], linewidth=2)
        ax1.plot(test_data.index, test_data.values, label='Actual', 
                color=self.colors['danger'], linewidth=2)
        ax1.plot(test_data.index, ma_pred, label='Moving Average', 
                color=self.colors['accent'], linewidth=2, linestyle='--')
        ax1.plot(test_data.index, trend_pred, label='Linear Trend', 
                color=self.colors['success'], linewidth=2, linestyle='--')
        
        ax1.axvline(x=train_data.index[-1], color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
        ax1.set_title('Demand Forecasting Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sales (Units)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Forecast errors
        ax2 = axes[0, 1]
        ma_error = np.abs(np.array(ma_pred) - test_data.values)
        trend_error = np.abs(trend_pred - test_data.values)
        
        ax2.plot(test_data.index, ma_error, label='Moving Average Error', 
                color=self.colors['accent'], linewidth=2)
        ax2.plot(test_data.index, trend_error, label='Linear Trend Error', 
                color=self.colors['success'], linewidth=2)
        ax2.set_title('Forecasting Errors Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        ax3 = axes[1, 0]
        ax3.hist(ma_error, bins=20, alpha=0.7, color=self.colors['accent'], 
                label=f'MA (MAPE: {np.mean(ma_error/test_data.values)*100:.1f}%)')
        ax3.hist(trend_error, bins=20, alpha=0.7, color=self.colors['success'], 
                label=f'Trend (MAPE: {np.mean(trend_error/test_data.values)*100:.1f}%)')
        ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Seasonal decomposition
        ax4 = axes[1, 1]
        # Simple seasonal decomposition
        seasonal_cycle = 365
        if len(train_data) >= seasonal_cycle:
            seasonal_component = []
            for i in range(min(seasonal_cycle, len(train_data))):
                seasonal_component.append(np.mean([train_data.iloc[j] 
                                                 for j in range(i, len(train_data), seasonal_cycle)]))
            
            ax4.plot(range(len(seasonal_component)), seasonal_component, 
                    color=self.colors['primary'], linewidth=2, marker='o', markersize=3)
            ax4.set_title('Seasonal Pattern (Daily)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Day of Year')
            ax4.set_ylabel('Average Sales')
            ax4.grid(True, alpha=0.3)
        
        # # Add mathematical equations as text boxes
        # equation_text = (
        #     r'ARIMA(p,d,q): $(1-\phi_1L-...-\phi_pL^p)(1-L)^d X_t = (1+\theta_1L+...+\theta_qL^q)\varepsilon_t$' + '\n' +
        #     r'LSTM: $h_t = f_t * h_{t-1} + i_t * \tilde{C}_t$' + '\n' +
        #     r'MAPE: $\frac{100}{n}\sum_{t=1}^{n}\left|\frac{A_t-F_t}{A_t}\right|$'
        # )
        
        # fig.text(0.02, 0.02, equation_text, fontsize=10, 
        #         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
        #         verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig('/Users/mayankdw/fast_delivery_paper/demand_forecasting_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_inventory_clustering_analysis(self):
        """
        Create K-means clustering analysis for inventory management.
        """
        # Prepare features for clustering
        product_features = self.sales_df.groupby(['Product', 'Category']).agg({
            'Sales': ['mean', 'std', 'sum'],
            'Date': 'count'
        }).round(2)
        
        product_features.columns = ['Avg_Sales', 'Sales_Std', 'Total_Sales', 'Days_Available']
        product_features = product_features.reset_index()
        
        # Add seasonality index
        seasonal_index = []
        for product in product_features['Product']:
            product_data = self.sales_df[self.sales_df['Product'] == product]
            monthly_avg = product_data.groupby('Month_Num')['Sales'].mean()
            seasonality = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() > 0 else 0
            seasonal_index.append(seasonality)
        
        product_features['Seasonality_Index'] = seasonal_index
        
        # Add profit margin (simulated)
        np.random.seed(self.random_state)
        product_features['Profit_Margin'] = np.random.uniform(0.1, 0.5, len(product_features))
        
        # Prepare features for clustering
        feature_cols = ['Avg_Sales', 'Sales_Std', 'Seasonality_Index', 'Profit_Margin']
        X = product_features[feature_cols]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        k = 4
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        product_features['Cluster'] = clusters
        
        # Create clustering visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Feature scatter plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(product_features['Avg_Sales'], product_features['Sales_Std'], 
                            c=clusters, cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Average Sales')
        ax1.set_ylabel('Sales Std Dev')
        ax1.set_title('Product Clustering: Avg Sales vs Variability', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Seasonality vs Profit Margin
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(product_features['Seasonality_Index'], product_features['Profit_Margin'], 
                             c=clusters, cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Seasonality Index')
        ax2.set_ylabel('Profit Margin')
        ax2.set_title('Product Clustering: Seasonality vs Profitability', fontsize=12, fontweight='bold')
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cluster characteristics
        ax3 = axes[0, 2]
        cluster_summary = product_features.groupby('Cluster')[feature_cols].mean()
        cluster_summary_scaled = pd.DataFrame(scaler.fit_transform(cluster_summary), 
                                            columns=cluster_summary.columns, 
                                            index=cluster_summary.index)
        
        im = ax3.imshow(cluster_summary_scaled.T, cmap='RdYlBu_r', aspect='auto')
        ax3.set_xticks(range(k))
        ax3.set_xticklabels([f'Cluster {i}' for i in range(k)])
        ax3.set_yticks(range(len(feature_cols)))
        ax3.set_yticklabels(feature_cols)
        ax3.set_title('Cluster Characteristics Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax3, label='Standardized Value')
        
        # Add text annotations
        for i in range(k):
            for j, feature in enumerate(feature_cols):
                text = ax3.text(i, j, f'{cluster_summary_scaled.iloc[j, i]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        # Plot 4: Cluster distribution
        ax4 = axes[1, 0]
        cluster_counts = product_features['Cluster'].value_counts().sort_index()
        bars = ax4.bar(range(k), cluster_counts.values, 
                      color=[plt.cm.tab10(i) for i in range(k)], alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Number of Products')
        ax4.set_title('Products per Cluster', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(k))
        ax4.set_xticklabels([f'Cluster {i}' for i in range(k)])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Category distribution by cluster
        ax5 = axes[1, 1]
        category_cluster = pd.crosstab(product_features['Category'], product_features['Cluster'])
        category_cluster.plot(kind='bar', stacked=True, ax=ax5, 
                            color=[plt.cm.tab10(i) for i in range(k)], alpha=0.7)
        ax5.set_title('Category Distribution by Cluster', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Category')
        ax5.set_ylabel('Number of Products')
        ax5.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Performance metrics by cluster
        ax6 = axes[1, 2]
        metrics = ['Avg_Sales', 'Total_Sales', 'Profit_Margin']
        cluster_performance = product_features.groupby('Cluster')[metrics].mean()
        
        x = np.arange(len(metrics))
        width = 0.2
        
        for i in range(k):
            offset = (i - k/2 + 0.5) * width
            bars = ax6.bar(x + offset, cluster_performance.iloc[i], width, 
                          label=f'Cluster {i}', alpha=0.7, color=plt.cm.tab10(i))
        
        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Average Value')
        ax6.set_title('Performance Metrics by Cluster', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add K-means equation
        equation_text = (
            r'K-means Objective: $J = \sum_{i=1}^{k} \sum_{p \in G_i} ||v(p) - c_i||^2$' + '\n' +
            r'where $v(p) = [\bar{D}_p, \sigma_{D_p}, S_p, P_p]^T$'
        )
        
        fig.text(0.02, 0.02, equation_text, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8),
                verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig('/Users/mayankdw/fast_delivery_paper/inventory_clustering_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return product_features
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        print("AI-Driven Demand Forecasting Analysis Summary")
        print("=" * 60)
        
        # Overall statistics
        total_products = self.sales_df['Product'].nunique()
        total_sales = self.sales_df['Sales'].sum()
        avg_daily_sales = self.sales_df.groupby('Date')['Sales'].sum().mean()
        
        print(f"Dataset Overview:")
        print(f"- Total Products: {total_products}")
        print(f"- Total Sales: {total_sales:,.0f} units")
        print(f"- Average Daily Sales: {avg_daily_sales:,.1f} units")
        print(f"- Date Range: {self.sales_df['Date'].min()} to {self.sales_df['Date'].max()}")
        
        # Category analysis
        print(f"\nCategory Performance:")
        category_stats = self.sales_df.groupby('Category')['Sales'].agg(['sum', 'mean', 'std']).round(1)
        print(category_stats)
        
        # Seasonal analysis
        print(f"\nSeasonal Patterns:")
        seasonal_stats = self.sales_df.groupby('Month')['Sales'].mean().round(1)
        peak_month = seasonal_stats.idxmax()
        low_month = seasonal_stats.idxmin()
        print(f"- Peak sales month: {peak_month} ({seasonal_stats[peak_month]:.1f} avg units)")
        print(f"- Lowest sales month: {low_month} ({seasonal_stats[low_month]:.1f} avg units)")
        
        # Save summary to file
        with open('/Users/mayankdw/fast_delivery_paper/demand_analysis_summary.txt', 'w') as f:
            f.write("AI-Driven Demand Forecasting Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset Overview:\n")
            f.write(f"- Total Products: {total_products}\n")
            f.write(f"- Total Sales: {total_sales:,.0f} units\n")
            f.write(f"- Average Daily Sales: {avg_daily_sales:,.1f} units\n")
            f.write(f"- Date Range: {self.sales_df['Date'].min()} to {self.sales_df['Date'].max()}\n\n")
            f.write("Category Performance:\n")
            f.write(category_stats.to_string())
            f.write(f"\n\nSeasonal Patterns:\n")
            f.write(f"- Peak sales month: {peak_month} ({seasonal_stats[peak_month]:.1f} avg units)\n")
            f.write(f"- Lowest sales month: {low_month} ({seasonal_stats[low_month]:.1f} avg units)\n")

def main():
    """Main function to run the complete AI demand forecasting analysis."""
    print("AI-Driven Demand Forecasting and Inventory Management Analysis")
    print("=" * 70)
    
    # Initialize analysis
    analysis = AIDemandforcestingAnalysis(random_state=42)
    
    # Generate synthetic sales data
    print("1. Generating synthetic sales data with seasonal patterns...")
    sales_data = analysis.generate_sales_data(years=3, products=50)
    
    # Create visualizations
    print("2. Creating sales distribution analysis...")
    analysis.create_sales_distribution_plots()
    
    print("3. Creating demand forecasting visualization...")
    analysis.create_demand_forecasting_visualization()
    
    print("4. Creating inventory clustering analysis...")
    product_features = analysis.create_inventory_clustering_analysis()
    
    print("5. Generating summary statistics...")
    analysis.generate_summary_statistics()
    
    # Save data
    sales_data.to_csv('/Users/mayankdw/fast_delivery_paper/synthetic_sales_data.csv', index=False)
    product_features.to_csv('/Users/mayankdw/fast_delivery_paper/product_clustering_results.csv', index=False)
    
    print(f"\nAll files saved to: /Users/mayankdw/fast_delivery_paper/")
    print("Generated files:")
    print("- sales_distribution_analysis.png")
    print("- demand_forecasting_analysis.png")
    print("- inventory_clustering_analysis.png")
    print("- synthetic_sales_data.csv")
    print("- product_clustering_results.csv")
    print("- demand_analysis_summary.txt")
    print("- ai_demand_prediction_equations.md")

if __name__ == "__main__":
    main()