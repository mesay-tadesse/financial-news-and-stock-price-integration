import pandas as pd


def read_csv_file(file_path):
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Remove any 'Unnamed:' columns
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

        # Get additional info
        column_names = data.columns.tolist()
        row_count = data.shape[0]

        return {"data": data, "column_names": column_names, "row_count": row_count}
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: The file at {file_path} could not be parsed.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def calculate_summary_statistics(data):
    """
    Calculate summary statistics for numerical columns in the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: A DataFrame containing the summary statistics.
    """
    return data.describe()


def calculate_correlation_matrix(data):
    """
    Calculate the correlation matrix for numerical columns in the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: A DataFrame containing the correlation matrix.
    """
    return data.corr()


def detect_outliers(data, column):
    """
    Detect outliers in a specified column using the IQR method.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to detect outliers for.

    Returns:
    pd.DataFrame: A DataFrame containing the rows with outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers


def convert_to_datetime(data, column):
    """
    Convert a specified column to datetime.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to convert to datetime.

    Returns:
    pd.DataFrame: The DataFrame with the converted column.
    """
    data[column] = pd.to_datetime(data[column])
    return data


def calculate_daily_returns(data, column):
    """
    Calculate daily returns for a specified column.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to calculate daily returns for.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for daily returns.
    """
    data["Daily_Returns"] = data[column].pct_change()
    return data


def calculate_moving_average(data, column, window):
    """
    Calculate moving average for a specified column.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to calculate moving average for.
    window (int): The window size for the moving average.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for moving average.
    """
    data[f"MA_{window}"] = data[column].rolling(window=window).mean()
    return data
