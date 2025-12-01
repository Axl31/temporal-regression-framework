"""Time-series windowing utilities.

convert_time_series creates sliding windows (t .. t+7) for all features and the
target. It renames non-date columns to variable1, variable2, ... and target -> 'target'.
It also returns a mapping dictionary to translate back to original names.
"""

import pandas as pd
import numpy as np

def convert_time_series(df, date_col, target_col, window_size=8):
    """Convert a DataFrame into sliding windows.

    Behavior:
        - Sort rows by date_col
        - Rename target_col -> 'target' and other feature columns to variable1..N
        - Create columns like variable1(t+0), variable1(t+1), ..., target(t+7)
        - Move all target(t+X) columns to the right (end)

    Returns:
        new_df (pd.DataFrame): windowed DataFrame
        mapping (dict): maps 'variable1' -> original_feature_name and 'target' -> original target
    """
    df = df.copy()

    # ensure date column is datetime-like
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(by=date_col).reset_index(drop=True)

    # Identify feature columns (exclude date and target)
    feature_cols = [c for c in df.columns if c not in [date_col, target_col]]

    # Build rename mapping and inverse mapping
    rename_map = {target_col: 'target'}
    mapping = {'target': target_col}
    for i, c in enumerate(feature_cols):
        new_name = f'variable{i+1}'
        rename_map[c] = new_name
        mapping[new_name] = c

    # Apply renaming
    df = df.rename(columns=rename_map)

    rows = []
    n = len(df)
    if n < window_size:
        raise ValueError(f'DataFrame has {n} rows but window_size is {window_size}')


    for i in range(n - window_size + 1):
        row = {}
        for col in df.columns:
            if col == date_col:
                continue
            for t in range(window_size):
                row[f"{col}(t+{t})"] = df.iloc[i + t][col]
        rows.append(row)

    new_df = pd.DataFrame(rows)

    # Move target(t+...) columns to the end in ascending order
    target_cols = sorted([c for c in new_df.columns if c.startswith('target(t+')],
                            key=lambda x: int(x.split('(t+')[1].rstrip(')')))
    other_cols = [c for c in new_df.columns if c not in target_cols]
    new_df = new_df[other_cols + target_cols]

    return new_df, mapping
