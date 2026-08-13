"""Tests for correlation-based association strategies."""

import pandas as pd
import pytest

from correlation_network.association._correlation import (
    KendallStrategy,
    PearsonStrategy,
    SpearmanStrategy,
)
from correlation_network.exceptions import InsufficientColumnsError


class TestPearsonStrategy:
    def test_returns_symmetric_matrix(self, numeric_dataframe: pd.DataFrame):
        strategy = PearsonStrategy()
        result = strategy.compute(numeric_dataframe)
        pd.testing.assert_frame_equal(result, result.T)

    def test_diagonal_is_one(self, numeric_dataframe: pd.DataFrame):
        strategy = PearsonStrategy()
        result = strategy.compute(numeric_dataframe)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_values_in_valid_range(self, numeric_dataframe: pd.DataFrame):
        strategy = PearsonStrategy()
        result = strategy.compute(numeric_dataframe)
        assert (result >= -1).all().all()
        assert (result <= 1).all().all()

    def test_detects_strong_positive_correlation(self, numeric_dataframe: pd.DataFrame):
        strategy = PearsonStrategy()
        result = strategy.compute(numeric_dataframe)
        assert result.loc["x", "y"] > 0.8

    def test_detects_negative_correlation(self, numeric_dataframe: pd.DataFrame):
        strategy = PearsonStrategy()
        result = strategy.compute(numeric_dataframe)
        assert result.loc["x", "w"] < -0.5


class TestSpearmanStrategy:
    def test_returns_symmetric_matrix(self, numeric_dataframe: pd.DataFrame):
        strategy = SpearmanStrategy()
        result = strategy.compute(numeric_dataframe)
        pd.testing.assert_frame_equal(result, result.T)

    def test_detects_strong_correlation(self, numeric_dataframe: pd.DataFrame):
        strategy = SpearmanStrategy()
        result = strategy.compute(numeric_dataframe)
        assert result.loc["x", "y"] > 0.7


class TestKendallStrategy:
    def test_returns_symmetric_matrix(self, numeric_dataframe: pd.DataFrame):
        strategy = KendallStrategy()
        result = strategy.compute(numeric_dataframe)
        pd.testing.assert_frame_equal(result, result.T)

    def test_detects_strong_correlation(self, numeric_dataframe: pd.DataFrame):
        strategy = KendallStrategy()
        result = strategy.compute(numeric_dataframe)
        assert result.loc["x", "y"] > 0.5


class TestInsufficientNumericColumns:
    @pytest.mark.parametrize(
        "strategy",
        [PearsonStrategy(), SpearmanStrategy(), KendallStrategy()],
        ids=["pearson", "spearman", "kendall"],
    )
    def test_categorical_only_frame_raises(
        self, strategy, categorical_dataframe: pd.DataFrame
    ):
        with pytest.raises(InsufficientColumnsError, match="at least 2 numeric"):
            strategy.compute(categorical_dataframe)

    def test_single_numeric_column_raises(self, numeric_dataframe: pd.DataFrame):
        with pytest.raises(InsufficientColumnsError) as excinfo:
            PearsonStrategy().compute(numeric_dataframe[["x"]])
        assert excinfo.value.found == ["x"]

    def test_error_names_the_offending_method(
        self, categorical_dataframe: pd.DataFrame
    ):
        with pytest.raises(InsufficientColumnsError) as excinfo:
            KendallStrategy().compute(categorical_dataframe)
        assert excinfo.value.method == "kendall"
