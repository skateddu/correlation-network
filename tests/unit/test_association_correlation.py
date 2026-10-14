"""Tests for correlation-based association strategies."""

import numpy as np
import pandas as pd
import pytest

from correlation_network.association._correlation import (
    KendallStrategy,
    PearsonStrategy,
    SpearmanStrategy,
)
from correlation_network.exceptions import (
    InsufficientColumnsError,
    NoCompleteObservationsError,
)


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


class TestCorrelationMissingValues:
    @pytest.fixture()
    def disjoint_dataframe(self) -> pd.DataFrame:
        """'p' and 'q' are never present on the same row; 'r' is complete."""
        return pd.DataFrame(
            {
                "p": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
                "q": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
                "r": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
            }
        )

    @pytest.mark.parametrize(
        "strategy",
        [PearsonStrategy(), SpearmanStrategy(), KendallStrategy()],
        ids=["pearson", "spearman", "kendall"],
    )
    def test_no_overlapping_rows_raises(
        self, strategy, disjoint_dataframe: pd.DataFrame
    ):
        # pandas would return NaN here, which thresholding turns into a silent
        # "no edge" instead of reporting an undefined association
        with pytest.raises(NoCompleteObservationsError) as excinfo:
            strategy.compute(disjoint_dataframe)
        assert set(excinfo.value.columns) == {"p", "q"}

    def test_partial_overlap_is_computed(self):
        df = pd.DataFrame(
            {
                "p": [1.0, 2.0, 3.0, 4.0, np.nan],
                "q": [2.0, 4.0, 6.0, 8.0, 1.0],
            }
        )
        result = PearsonStrategy().compute(df)
        assert result.loc["p", "q"] == pytest.approx(1.0)
