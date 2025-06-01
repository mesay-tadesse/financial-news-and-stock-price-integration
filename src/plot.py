import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(data, column, bins=10):
    """
    Plots a histogram for a specified column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to plot the histogram for.
    bins (int): The number of bins for the histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, edgecolor="k")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_scatter(data, column1, column2):
    """
    Plots a scatter plot for two specified columns in the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column1 (str): The column name for the x-axis.
    column2 (str): The column name for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[column1], data[column2], alpha=0.5)
    plt.title(f"Scatter Plot of {column1} vs {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()


def plot_time_series(data, date_column, value_column):
    """
    Plots a time series line chart for a specified date and value column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    date_column (str): The column name for the dates.
    value_column (str): The column name for the values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data[date_column], data[value_column])
    plt.title(f"Time Series of {value_column}")
    plt.xlabel("Date")
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.show()


def plot_boxplot(data, column):
    """
    Plots a box plot for a specified column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to plot the box plot for.
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(data[column])
    plt.title(f"Box Plot of {column}")
    plt.ylabel(column)
    plt.show()


def plot_correlation_matrix(data):
    """
    Plots a correlation matrix heatmap for the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def plot_sentiment_distribution(data, column):
    """
    Plots the sentiment distribution as a bar chart.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name for sentiment labels.
    """
    sentiment_counts = data[column].value_counts()
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()


def plot_average_sentiment_by_publisher(data, publisher_column, sentiment_column):
    """
    Plots the average sentiment score by publisher as a horizontal bar chart.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    publisher_column (str): The column name for publishers.
    sentiment_column (str): The column name for sentiment scores.
    """
    average_sentiment_by_publisher = data.groupby(publisher_column)[sentiment_column].mean()
    average_sentiment_by_publisher.sort_values().plot(kind='barh', figsize=(10, 6), color='purple')
    plt.title('Average Sentiment Score by Publisher')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Publisher')
    plt.show()


def plot_sentiment_over_time(data, date_column, sentiment_column):
    """
    Plots the average sentiment score over time as a line chart.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    date_column (str): The column name for dates.
    sentiment_column (str): The column name for sentiment scores.
    """
    data[date_column] = pd.to_datetime(data[date_column])  # Ensure date is in datetime format
    data.set_index(date_column, inplace=True)
    sentiment_over_time = data.resample('M')[sentiment_column].mean()
    plt.figure(figsize=(12, 6))
    sentiment_over_time.plot(color='orange')
    plt.title('Average Sentiment Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.show()
