"""
Data loader utilities.

Responsible for reading CSV/XLSX files, preprocessing, and optionally
converting time series into sliding windows.
"""

import pandas as pd
from utils.preprocessing import drop_rows_with_zeros, ensure_datetime
from utils.time_series import convert_time_series

def load_file(filename):
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Args:
        filename (str): Path to .csv or .xlsx/.xls file.

    Returns:
        pd.DataFrame: Loaded data
    """
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filename)
    else:
        raise ValueError("File must be .csv or .xlsx/.xls")

    # Drop a common index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    return df


def load_data(filename, date_col, target_col, columns_to_exclude=None, window_size=8, split=True, test_size=0.2):
    """
    Load and preprocess data, optionally generating time-series windows
    and splitting into train/test sets.

    Args:
        filename (str): Path to data file.
        date_col (str): Column name containing dates.
        target_col (str): Column name to predict.
        columns_to_exclude (list, optional): Columns to exclude from zero-checking.
        window_size (int): Sliding window size for time-series transformation.
        split (bool): Whether to split into train/test.
        test_size (float or int): Fraction or number of test samples.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, mapping)
    """
    # 1. Load raw data
    df = load_file(filename)

    # 2. Ensure date column is datetime
    df = ensure_datetime(df, date_col)

    # 3. Drop rows with zeros in selected columns
    df = drop_rows_with_zeros(df, exclude_cols=columns_to_exclude)

    # 4. Convert to time-series sliding windows
    df_windows, mapping = convert_time_series(df, date_col=date_col, target_col=target_col, window_size=window_size)

    # 5. Separate features and target
    target_cols = [f"target(t+{window_size-1})"]
    feature_cols = [col for col in df_windows.columns if col not in target_cols]

    X = df_windows[feature_cols]
    y = df_windows[target_cols]

    # 6. Split into train/test if required
    if split:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    else:
        # manual split
        if test_size < 1:
            test_size = int(len(df_windows) * test_size)
        split_idx = len(df_windows) - test_size
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, mapping
