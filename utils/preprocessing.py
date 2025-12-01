"""Preprocessing helpers.

This module contains utilities to clean the dataframe before windowing,
for example removing rows with zeros in selected columns, or converting
date columns to datetime.
"""

import pandas as pd
import numpy as np

def ensure_datetime(df, date_col):
    """Ensure the specified column is datetime and return a copy of df.

    This avoids in-place changes to user dataframes.
    """
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    return df

def drop_rows_with_zeros(df, exclude_cols=None):
    """Remove rows that have zero in any column except those excluded.

    Args:
    df (pd.DataFrame)
    exclude_cols (list or None): columns to exclude from zero-checking
    date_col (str or None): name of the date column (also excluded)


    Returns:
    pd.DataFrame
    """
    if exclude_cols is None:
        exclude_cols = []
    exclude = set(exclude_cols)

    cols_to_check = df.columns.difference(exclude_cols)
    if cols_to_check.empty:
        return df.copy()

    return df[cols_to_check]
