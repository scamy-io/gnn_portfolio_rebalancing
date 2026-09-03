import pandas as pd

from src.config import HORIZON


def compute_forward_returns(close_wide: pd.DataFrame,
                            horizon: int = HORIZON) -> pd.DataFrame:
    if not close_wide.index.is_monotonic_increasing:
        close_wide = close_wide.sort_index()
    return close_wide.shift(-horizon) / close_wide - 1
