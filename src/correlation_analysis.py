"""
Utility functions for sentiment analysis of news headlines.
"""
from textblob import TextBlob
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Union, Optional, Tuple

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
                              symbol_column_name: Optional[str] = None) -> pd.DataFrame:
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
        print(f"Warning: Symbol column '{symbol_column_name}' not found in DataFrame. Aggregating sentiment generally by date only.")
        symbol_column_name = None # Proceed with general aggregation

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

    return daily_sentiment
def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate daily returns from price series.

    Args:
        prices (pd.Series): Series of stock prices

    Returns:
        pd.Series: Daily returns as percentage changes
    """
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.pct_change()
def align_sentiment_returns(sentiment_df: pd.DataFrame,
                          returns_df: pd.DataFrame,
                          sentiment_date_col: str,
                          returns_date_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align sentiment and returns data by date.

    Args:
        sentiment_df (pd.DataFrame): DataFrame containing sentiment data with columns:
                                   - sentiment_date_col: Date column
                                   - 'mean_polarity': Mean sentiment polarity
                                   - 'std_polarity': Standard deviation of polarity
                                   - 'headline_count': Number of headlines
                                   - 'mean_subjectivity': Mean subjectivity score
        returns_df (pd.DataFrame): DataFrame containing returns data with columns:
                                 - returns_date_col: Date column
                                 - 'returns': Calculated stock returns
        sentiment_date_col (str): Name of date column in sentiment_df
        returns_date_col (str): Name of date column in returns_df

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Aligned sentiment and returns DataFrames

    Raises:
        ValueError: If required columns are missing, date parsing fails, or no common dates.
    """
    if sentiment_df.empty or returns_df.empty:
        print("Warning: One or both DataFrames for alignment are empty.")
        return pd.DataFrame(), pd.DataFrame()

    required_sentiment_cols = {sentiment_date_col, 'mean_polarity', 'std_polarity',
                             'headline_count', 'mean_subjectivity'}
    missing_sentiment_cols = required_sentiment_cols - set(sentiment_df.columns)
    if missing_sentiment_cols:
        raise ValueError(f"Missing required columns in sentiment_df: {missing_sentiment_cols}")

    required_returns_cols = {returns_date_col, 'returns'}
    missing_returns_cols = required_returns_cols - set(returns_df.columns)
    if missing_returns_cols:
        raise ValueError(f"Missing required columns in returns_df: {missing_returns_cols}")

    sentiment_df_copy = sentiment_df.copy()
    returns_df_copy = returns_df.copy()

    try:
        sentiment_df_copy[sentiment_date_col] = pd.to_datetime(sentiment_df_copy[sentiment_date_col])
        returns_df_copy[returns_date_col] = pd.to_datetime(returns_df_copy[returns_date_col])
    except Exception as e:
        raise ValueError(f"Failed to parse date columns: {str(e)}")

    sentiment_df_copy = sentiment_df_copy.set_index(sentiment_date_col).sort_index()
    returns_df_copy = returns_df_copy.set_index(returns_date_col).sort_index()

    common_dates = sentiment_df_copy.index.intersection(returns_df_copy.index)
    if len(common_dates) == 0:
        # This is a valid scenario if a stock has no news or vice-versa on common days
        # print(f"Warning: No common dates found between sentiment and returns data for the current pair.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs

    aligned_sentiment = sentiment_df_copy.loc[common_dates]
    aligned_returns = returns_df_copy.loc[common_dates]

    if aligned_sentiment.empty or aligned_returns.empty:
        print("Warning: Alignment resulted in empty DataFrames, though common dates were found. Check data.")
        return pd.DataFrame(), pd.DataFrame()

    # print(f"Aligned {len(common_dates)} common dates. Date range: {common_dates.min().date()} to {common_dates.max().date()}")
    return aligned_sentiment, aligned_returns

def calculate_correlation_metrics(sentiment_series: pd.Series,
                               returns_series: pd.Series) -> Dict[str, float]:
    """
    Calculate various correlation metrics between sentiment and returns.
    
    Args:
        sentiment_series (pd.Series): Series of sentiment scores
        returns_series (pd.Series): Series of stock returns
        
    Returns:
        Dict[str, float]: Dictionary containing correlation metrics. Returns None for all metrics if insufficient data.
        
    Raises:
        ValueError: If input series have different lengths
    """
    # Check if series have the same length
    if len(sentiment_series) != len(returns_series):
        raise ValueError("Input series must have the same length")
    
    # Check if we have enough data points
    if len(sentiment_series) < 2:
        return {
            'pearson_correlation': None,
            'pearson_p_value': None,
            'spearman_correlation': None,
            'spearman_p_value': None,
            'valid': False,
            'n_observations': len(sentiment_series)
        }
    
    try:
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(sentiment_series, returns_series)
        
        # Spearman rank correlation (handles ties automatically)
        spearman_corr, spearman_p = stats.spearmanr(sentiment_series, returns_series)
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'valid': True,
            'n_observations': len(sentiment_series)
        }
    except (ValueError, RuntimeWarning) as e:
        # Handle cases where correlation cannot be calculated
        print(f"Warning: Could not calculate correlation - {str(e)}")
        return {
            'pearson_correlation': None,
            'pearson_p_value': None,
            'spearman_correlation': None,
            'spearman_p_value': None,
            'valid': False,
            'n_observations': len(sentiment_series),
            'error': str(e)
        }

def analyze_lagged_correlations(sentiment_series: pd.Series,
                              returns_series: pd.Series,
                              max_lag: int = 5) -> pd.DataFrame:
    """
    Analyze correlations with different lags between sentiment and returns.
    
    Args:
        sentiment_series (pd.Series): Series of sentiment scores with datetime index
        returns_series (pd.Series): Series of stock returns with datetime index
        max_lag (int): Maximum number of lags to analyze in each direction
        
    Returns:
        pd.DataFrame: DataFrame containing correlation metrics for each lag
        
    Raises:
        ValueError: If input series have different lengths or invalid data
    """
    if len(sentiment_series) != len(returns_series):
        raise ValueError("Input series must have the same length")
    
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    
    results = []
    
    # Ensure we're working with pandas Series with datetime index
    if not isinstance(sentiment_series.index, pd.DatetimeIndex):
        sentiment_series = pd.Series(sentiment_series.values, index=pd.to_datetime(sentiment_series.index))
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        returns_series = pd.Series(returns_series.values, index=pd.to_datetime(returns_series.index))
    
    # Calculate correlation for each lag
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Negative lag: sentiment leads returns
            shifted_sentiment = sentiment_series.shift(-lag)
            shifted_returns = returns_series
            direction = 'sentiment_lead'
        elif lag > 0:
            # Positive lag: returns lead sentiment
            shifted_sentiment = sentiment_series
            shifted_returns = returns_series.shift(lag)
            direction = 'returns_lead'
        else:
            # No lag
            shifted_sentiment = sentiment_series
            shifted_returns = returns_series
            direction = 'same_day'
        
        # Remove NA values from shifted series
        valid_indices = ~(shifted_sentiment.isna() | shifted_returns.isna())
        n_valid = valid_indices.sum()
        
        if n_valid < 2:  # Need at least 2 points for correlation
            print(f"Warning: Insufficient data points (n={n_valid}) for lag {lag}")
            continue
            
        # Calculate metrics for this lag
        metrics = calculate_correlation_metrics(
            shifted_sentiment[valid_indices],
            shifted_returns[valid_indices]
        )
        
        # Add lag information
        metrics['lag'] = lag
        metrics['direction'] = direction
        metrics['start_date'] = shifted_sentiment[valid_indices].index.min()
        metrics['end_date'] = shifted_sentiment[valid_indices].index.max()
        metrics['n_observations'] = n_valid
        
        results.append(metrics)
    
    if not results:
        print("Warning: No valid lag correlations could be calculated")
        return pd.DataFrame()
    
    # Convert results to DataFrame and sort by lag
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    col_order = ['lag', 'direction', 'n_observations', 'start_date', 'end_date',
                 'pearson_correlation', 'pearson_p_value',
                 'spearman_correlation', 'spearman_p_value', 'valid']
    
    # Only include columns that exist in the results
    col_order = [col for col in col_order if col in df.columns]
    
    return df[col_order].sort_values('lag')