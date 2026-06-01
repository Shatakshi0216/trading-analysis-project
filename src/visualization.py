"""
Visualization module for trading analysis project.
Creates professional visualizations for data analysis and insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional


class TradingVisualizer:
    """Create visualizations for trading analysis."""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer with style."""
        plt.style.use(style)
        sns.set_palette("husl")
    
    @staticmethod
    def plot_sentiment_trends(df: pd.DataFrame, figsize=(14, 6)):
        """Plot sentiment trends over time."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(df['date'], df['fear_greed_value'], linewidth=2, label='Fear/Greed Index')
        ax.fill_between(df['date'], df['fear_greed_value'], alpha=0.3)
        ax.axhline(y=45, color='red', linestyle='--', alpha=0.5, label='Fear Threshold')
        ax.axhline(y=55, color='green', linestyle='--', alpha=0.5, label='Greed Threshold')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Fear/Greed Value', fontsize=12)
        ax.set_title('Bitcoin Market Sentiment Trends', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pnl_distribution(df: pd.DataFrame, figsize=(14, 6)):
        """Plot PnL distribution."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(df['pnl'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(df['pnl'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["pnl"].mean():.2f}')
        axes[0].set_xlabel('PnL', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('PnL Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by sentiment
        if 'sentiment_category' in df.columns:
            df.boxplot(column='pnl', by='sentiment_category', ax=axes[1])
            axes[1].set_xlabel('Sentiment Category', fontsize=12)
            axes[1].set_ylabel('PnL', fontsize=12)
            axes[1].set_title('PnL by Sentiment Category', fontsize=14, fontweight='bold')
            plt.sca(axes[1])
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, figsize=(10, 8)):
        """Plot correlation heatmap."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Correlation Matrix: Sentiment & Trading Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_performance_metrics(df: pd.DataFrame, figsize=(14, 6)):
        """Plot key performance metrics."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Win rate by leverage
        if 'leverage_category' in df.columns:
            leverage_perf = df.groupby('leverage_category').agg({
                'is_profitable': 'mean',
                'pnl': 'mean'
            }).reset_index()
            
            axes[0].bar(leverage_perf['leverage_category'], leverage_perf['is_profitable'] * 100)
            axes[0].set_xlabel('Leverage Category', fontsize=12)
            axes[0].set_ylabel('Win Rate (%)', fontsize=12)
            axes[0].set_title('Win Rate by Leverage', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')
        
        # Cumulative PnL
        if 'cumulative_pnl' in df.columns:
            df_sorted = df.sort_values('timestamp')
            axes[1].plot(df_sorted.index, df_sorted['cumulative_pnl'], linewidth=2)
            axes[1].fill_between(df_sorted.index, df_sorted['cumulative_pnl'], alpha=0.3)
            axes[1].set_xlabel('Trade Number', fontsize=12)
            axes[1].set_ylabel('Cumulative PnL', fontsize=12)
            axes[1].set_title('Cumulative PnL Over Time', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sentiment_vs_performance(df: pd.DataFrame, figsize=(12, 6)):
        """Plot sentiment vs trading performance scatter."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'fear_greed_value' in df.columns:
            scatter = ax.scatter(df['fear_greed_value'], df['pnl'], 
                               c=df['pnl'], cmap='RdYlGn', alpha=0.6, s=50)
            
            # Add trend line
            z = np.polyfit(df['fear_greed_value'].dropna(), df['pnl'].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['fear_greed_value'].min(), df['fear_greed_value'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
            
            ax.set_xlabel('Fear/Greed Index', fontsize=12)
            ax.set_ylabel('PnL', fontsize=12)
            ax.set_title('Sentiment vs Trading Performance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('PnL', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_hourly_patterns(df: pd.DataFrame, figsize=(14, 6)):
        """Plot hourly trading patterns."""
        if 'hour' in df.columns:
            hourly = df.groupby('hour').agg({'pnl': 'mean', 'is_profitable': 'mean'}).reset_index()
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            axes[0].bar(hourly['hour'], hourly['pnl'], color='skyblue', edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Hour of Day', fontsize=12)
            axes[0].set_ylabel('Average PnL', fontsize=12)
            axes[0].set_title('Average PnL by Hour', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            axes[1].bar(hourly['hour'], hourly['is_profitable'] * 100, color='lightgreen', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Hour of Day', fontsize=12)
            axes[1].set_ylabel('Win Rate (%)', fontsize=12)
            axes[1].set_title('Win Rate by Hour', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            return fig
        
        return None
    
    @staticmethod
    def create_interactive_dashboard(df: pd.DataFrame) -> go.Figure:
        """Create interactive Plotly dashboard."""
        fig = go.Figure()
        
        # Time series
        if 'fear_greed_value' in df.columns and 'timestamp' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['fear_greed_value'],
                mode='lines+markers',
                name='Fear/Greed Index',
                hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Trading Analysis Interactive Dashboard',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
