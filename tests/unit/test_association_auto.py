"""Tests for the auto-detection association strategy."""

import pandas as pd
import pytest

from correlation_network.association._auto import AutoAssociationStrategy


class TestAutoAssociationStrategy:
    def test_numeric_only(self, numeric_dataframe: pd.DataFrame):
        strategy = AutoAssociationStrategy()
        result = strategy.compute(numeric_dataframe)
        assert result.shape == (4, 4)
        assert set(result.columns) == {"x", "y", "z", "w"}

    def test_categorical_only(self, categorical_dataframe: pd.DataFrame):
        strategy = AutoAssociationStrategy()
        result = strategy.compute(categorical_dataframe)
        assert result.shape == (3, 3)

    def test_mixed_types(self, mixed_dataframe: pd.DataFrame):
        strategy = AutoAssociationStrategy()
        result = strategy.compute(mixed_dataframe)
        assert "category" in result.columns
        assert "value" in result.columns
        assert "noise" in result.columns

    def test_returns_symmetric_matrix(self, mixed_dataframe: pd.DataFrame):
        strategy = AutoAssociationStrategy()
        result = strategy.compute(mixed_dataframe)
        pd.testing.assert_frame_equal(result, result.T, atol=1e-10)

    def test_diagonal_is_one(self, mixed_dataframe: pd.DataFrame):
        strategy = AutoAssociationStrategy()
        result = strategy.compute(mixed_dataframe)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_values_in_valid_range(self, mixed_dataframe: pd.DataFrame):
        strategy = AutoAssociationStrategy()
        result = strategy.compute(mixed_dataframe)
        # Auto mode uses abs values for numeric, so all should be [0, 1]
        assert (result >= -0.01).all().all()
        assert (result <= 1.01).all().all()
