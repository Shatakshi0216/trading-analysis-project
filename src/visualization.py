"""Visualization module for trading analysis project."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

class TradingVisualizer:
    """Create visualizations for trading analysis."""
    def __init__(self):
        try:
            plt.style.use('default')
        except:
            pass
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
    def plot_sentiment_vs_performance(df: pd.DataFrame, figsize=(12, 6)):
        """Plot sentiment vs trading performance."""
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(df['fear_greed_value'], df['pnl'], c=df['pnl'], cmap='RdYlGn', alpha=0.6, s=50)
        valid_idx = ~(df['fear_greed_value'].isna() | df['pnl'].isna())
        if valid_idx.sum() > 1:
            z = np.polyfit(df.loc[valid_idx, 'fear_greed_value'], df.loc[valid_idx, 'pnl'], 1)
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