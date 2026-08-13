"""Shared test fixtures."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def numeric_dataframe() -> pd.DataFrame:
    """DataFrame with only numeric columns and known correlations."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    y = x * 0.8 + rng.normal(0, 0.3, n)  # strongly correlated with x
    z = rng.normal(0, 1, n)  # independent
    w = -x * 0.7 + rng.normal(0, 0.4, n)  # negatively correlated with x
    return pd.DataFrame({"x": x, "y": y, "z": z, "w": w})


@pytest.fixture()
def categorical_dataframe() -> pd.DataFrame:
    """DataFrame with only categorical columns."""
    rng = np.random.default_rng(42)
    n = 300
    color = rng.choice(["red", "green", "blue"], n)
    # size is correlated with color
    size_map = {"red": ["S", "M"], "green": ["M", "L"], "blue": ["L", "XL"]}
    size = [rng.choice(size_map[c]) for c in color]
    shape = rng.choice(["circle", "square", "triangle"], n)  # independent
    return pd.DataFrame({"color": color, "size": size, "shape": shape})


@pytest.fixture()
def mixed_dataframe() -> pd.DataFrame:
    """DataFrame with both numeric and categorical columns."""
    rng = np.random.default_rng(42)
    n = 300
    category = rng.choice(["A", "B", "C"], n)
    value = np.where(
        np.array(category) == "A",
        rng.normal(10, 1, n),
        np.where(np.array(category) == "B", rng.normal(20, 1, n), rng.normal(30, 1, n)),
    )
    noise = rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "category": category,
            "value": value,
            "noise": noise,
        }
    )


@pytest.fixture()
def dataframe_with_id(numeric_dataframe: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with an integer ID column and a string UUID-like column."""
    n = len(numeric_dataframe)
    df = numeric_dataframe.copy()
    df["row_id"] = range(n)
    df["uuid"] = [f"id-{i:06d}" for i in range(n)]
    return df


@pytest.fixture()
def dataframe_with_datetime(numeric_dataframe: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with a datetime column."""
    n = len(numeric_dataframe)
    df = numeric_dataframe.copy()
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=n, freq="h")
    return df
