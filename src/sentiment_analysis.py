"""
Utility functions for sentiment analysis of news headlines.
"""
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime, timedelta

def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze the sentiment of a given text using TextBlob.

    Args:
        text (str): Input text for sentiment analysis

    Returns:
        Dict[str, float]: Dictionary containing polarity and subjectivity scores
    """
    analysis = TextBlob(str(text)) # Ensure text is string
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def get_sentiment_label(polarity: float) -> str:
    """
    Convert polarity score to sentiment label.

    Args:
        polarity (float): Sentiment polarity score

    Returns:
        str: Sentiment label ('positive', 'negative', or 'neutral')
    """
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    return 'neutral'

def analyze_headlines(headlines: List[str]) -> pd.DataFrame:
    """
    Analyze sentiment for a list of headlines.

    Args:
        headlines (List[str]): List of news headlines

    Returns:
        pd.DataFrame: DataFrame with sentiment scores and labels
    """
    results = []
    for headline in headlines:
        if pd.isna(headline): # Handle potential NaN headlines
            sentiment = {'polarity': 0.0, 'subjectivity': 0.0, 'text': '', 'label': 'neutral'}
        else:
            sentiment = analyze_sentiment(headline)
            sentiment['text'] = headline
            sentiment['label'] = get_sentiment_label(sentiment['polarity'])
        results.append(sentiment)

    return pd.DataFrame(results)

def aggregate_daily_sentiment(df: pd.DataFrame,
                              date_column: str,
                              symbol_column_name: Optional[str] = None,
                              min_headlines: int = 1) -> pd.DataFrame:
    """
    Aggregate sentiment scores by date, and optionally by symbol.

    Args:
        df (pd.DataFrame): DataFrame containing sentiment scores and dates.
                           Must include 'polarity', 'subjectivity', and the date_column.
                           May include the symbol_column_name.
        date_column (str): Name of the date column.
        symbol_column_name (Optional[str]): Name of the stock symbol column, if available
                                           and stock-specific aggregation is desired.

    Returns:
        pd.DataFrame: DataFrame with daily (and optionally per-symbol) aggregated sentiment scores.
                      Columns will include the date_column, (optionally symbol_column_name),
                      'mean_polarity', 'std_polarity', 'headline_count', 'mean_subjectivity'.

    Raises:
        ValueError: If required columns are missing or date parsing fails.
    """
    required_cols = {'polarity', 'subjectivity', date_column}
    if symbol_column_name and symbol_column_name not in df.columns:
        available_cols = ', '.join(df.columns)
        print(f"Warning: Symbol column '{symbol_column_name}' not found. Available columns: {available_cols}")
        print("Proceeding with market-wide sentiment aggregation.")
        symbol_column_name = None  # Proceed with general aggregation
    
    # Convert date column to datetime if it isn't already
    df[date_column] = pd.to_datetime(df[date_column])

    if symbol_column_name and symbol_column_name in df.columns:
        required_cols.add(symbol_column_name)

    missing_columns = required_cols - set(df.columns)
    # We only raise error for core sentiment/date columns, symbol is optional for this function to proceed
    core_missing_cols = {'polarity', 'subjectivity', date_column} - set(df.columns)
    if core_missing_cols:
        raise ValueError(f"Missing required core columns: {core_missing_cols}")

    try:
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    except Exception as e:
        raise ValueError(f"Failed to parse date column '{date_column}': {str(e)}")

    group_by_cols = [date_column]
    if symbol_column_name and symbol_column_name in df_copy.columns:
        group_by_cols.append(symbol_column_name)

    daily_sentiment = df_copy.groupby(group_by_cols).agg(
        mean_polarity=('polarity', 'mean'),
        std_polarity=('polarity', 'std'),
        headline_count=('polarity', 'count'),
        mean_subjectivity=('subjectivity', 'mean')
    ).reset_index()

    # Fill NaN std_polarity with 0 (occurs when there's only one headline for a date/symbol)
    daily_sentiment['std_polarity'] = daily_sentiment['std_polarity'].fillna(0)

    numeric_cols = ['mean_polarity', 'std_polarity', 'headline_count', 'mean_subjectivity']
    for col in numeric_cols:
        if col in daily_sentiment.columns: # Ensure column exists before astype
            daily_sentiment[col] = daily_sentiment[col].astype(float)

    # Filter out dates with insufficient headlines if specified
    if min_headlines > 1:
        daily_sentiment = daily_sentiment[daily_sentiment['headline_count'] >= min_headlines]
        if daily_sentiment.empty:
            print(f"Warning: No dates found with at least {min_headlines} headlines")
    
    return daily_sentiment

def analyze_sentiment_coverage(sentiment_df: pd.DataFrame, date_column: str) -> Dict[str, Union[int, float, str]]:
    """
    Analyze the coverage and quality of sentiment data.
    
    Args:
        sentiment_df: DataFrame containing sentiment data
        date_column: Name of the date column
        
    Returns:
        Dict containing coverage metrics
    """
    if sentiment_df.empty:
        return {
            'total_days': 0,
            'days_with_sentiment': 0,
            'coverage_percentage': 0.0,
            'avg_headlines_per_day': 0.0,
            'date_range': 'No data'
        }
    
    # Ensure date column is datetime
    sentiment_df[date_column] = pd.to_datetime(sentiment_df[date_column])
    
    # Calculate date range
    start_date = sentiment_df[date_column].min()
    end_date = sentiment_df[date_column].max()
    total_days = (end_date - start_date).days + 1
    
    # Calculate coverage
    days_with_sentiment = sentiment_df[date_column].nunique()
    coverage_percentage = (days_with_sentiment / total_days) * 100
    
    # Calculate average headlines per day
    avg_headlines = sentiment_df['headline_count'].mean() if 'headline_count' in sentiment_df.columns else None
    
    return {
        'total_days': total_days,
        'days_with_sentiment': days_with_sentiment,
        'coverage_percentage': coverage_percentage,
        'avg_headlines_per_day': avg_headlines,
        'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    }

def plot_sentiment_distribution(sentiment_df: pd.DataFrame, stock_name: str = None) -> plt.Figure:
    """
    Create a distribution plot of sentiment polarity scores.
    
    Args:
        sentiment_df: DataFrame containing sentiment data
        stock_name: Optional stock name for the title
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram of polarities
    sns.histplot(data=sentiment_df, x='mean_polarity', bins=30, ax=ax1)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Distribution of Sentiment Polarity' + (f' for {stock_name}' if stock_name else ''))
    ax1.set_xlabel('Sentiment Polarity')
    ax1.set_ylabel('Count')
    
    # Box plot of polarities by sentiment label
    if 'label' in sentiment_df.columns:
        sns.boxplot(data=sentiment_df, x='label', y='mean_polarity', ax=ax2)
        ax2.set_title('Sentiment Polarity by Label')
    else:
        sentiment_df['sentiment_category'] = pd.cut(
            sentiment_df['mean_polarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        sns.boxplot(data=sentiment_df, x='sentiment_category', y='mean_polarity', ax=ax2)
        ax2.set_title('Sentiment Polarity by Category')
    
    plt.tight_layout()
    return fig

def plot_sentiment_trends(sentiment_df: pd.DataFrame, 
                         date_column: str,
                         window: int = 7,
                         stock_name: str = None) -> plt.Figure:
    """
    Plot sentiment trends over time with rolling statistics.
    
    Args:
        sentiment_df: DataFrame containing sentiment data
        date_column: Name of the date column
        window: Rolling window size in days
        stock_name: Optional stock name for the title
        
    Returns:
        matplotlib Figure object
    """
    # Ensure we have datetime index
    df = sentiment_df.set_index(date_column) if date_column in sentiment_df.columns else sentiment_df
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Plot daily sentiment and rolling mean
    df['mean_polarity'].plot(marker='.', alpha=0.5, linestyle='none', ax=ax1, label='Daily')
    df['mean_polarity'].rolling(window=window).mean().plot(
        ax=ax1, linewidth=2, label=f'{window}-day Moving Average'
    )
    
    # Add confidence bands (rolling mean ± rolling std)
    rolling_mean = df['mean_polarity'].rolling(window=window).mean()
    rolling_std = df['mean_polarity'].rolling(window=window).std()
    ax1.fill_between(
        df.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.2,
        label='±1 Std. Dev.'
    )
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax1.set_title('Sentiment Trend' + (f' for {stock_name}' if stock_name else ''))
    ax1.set_ylabel('Sentiment Polarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot headline count
    if 'headline_count' in df.columns:
        df['headline_count'].plot(kind='bar', ax=ax2, alpha=0.5)
        ax2.set_title('Daily Headline Count')
        ax2.set_ylabel('Number of Headlines')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig
def plot_sentiment_returns_timeseries(sentiment_df, returns_df, stock_name):
    """
    Create a dual-axis time series plot showing sentiment and returns over time.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot sentiment on left axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Mean Polarity', color=color)
    ax1.plot(sentiment_df.index, sentiment_df['mean_polarity'], color=color, label='Sentiment')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add sentiment moving average
    sentiment_ma = sentiment_df['mean_polarity'].rolling(window=5).mean()
    ax1.plot(sentiment_df.index, sentiment_ma, color='darkblue', linestyle='--', 
             label='Sentiment 5-day MA', alpha=0.7)
    
    # Create second y-axis for returns
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Returns (%)', color=color)
    ax2.plot(returns_df.index, returns_df['returns'] * 100, color=color, label='Returns')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add returns moving average
    returns_ma = returns_df['returns'].rolling(window=5).mean() * 100
    ax2.plot(returns_df.index, returns_ma, color='darkred', linestyle='--', 
             label='Returns 5-day MA', alpha=0.7)
    
    # Add title and grid
    plt.title(f'Sentiment and Returns Time Series for {stock_name}')
    ax1.grid(True, alpha=0.3)
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    return fig