"""Tests for categorical association strategies."""

import numpy as np
import pandas as pd
import pytest

from correlation_network.association._categorical import (
    CorrelationRatioStrategy,
    CramersVStrategy,
)
from correlation_network.exceptions import (
    InsufficientColumnsError,
    NoCompleteObservationsError,
)


class TestCramersVStrategy:
    def test_returns_symmetric_matrix(self, categorical_dataframe: pd.DataFrame):
        strategy = CramersVStrategy()
        result = strategy.compute(categorical_dataframe)
        pd.testing.assert_frame_equal(result, result.T, atol=1e-10)

    def test_diagonal_is_one(self, categorical_dataframe: pd.DataFrame):
        strategy = CramersVStrategy()
        result = strategy.compute(categorical_dataframe)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_values_in_valid_range(self, categorical_dataframe: pd.DataFrame):
        strategy = CramersVStrategy()
        result = strategy.compute(categorical_dataframe)
        assert (result >= 0).all().all()
        assert (result <= 1).all().all()

    def test_detects_association_between_correlated_columns(
        self, categorical_dataframe: pd.DataFrame
    ):
        strategy = CramersVStrategy()
        result = strategy.compute(categorical_dataframe)
        # color and size are correlated by construction
        assert result.loc["color", "size"] > result.loc["color", "shape"]

    def test_only_uses_categorical_columns(self, categorical_dataframe: pd.DataFrame):
        strategy = CramersVStrategy()
        df = categorical_dataframe.assign(value=range(len(categorical_dataframe)))
        result = strategy.compute(df)
        assert "value" not in result.columns
        assert "color" in result.columns

    def test_too_few_categorical_columns_raises(self, mixed_dataframe: pd.DataFrame):
        # mixed_dataframe has a single categorical column ("category")
        strategy = CramersVStrategy()
        with pytest.raises(InsufficientColumnsError, match="at least 2 categorical"):
            strategy.compute(mixed_dataframe)

    def test_numeric_only_frame_raises(self, numeric_dataframe: pd.DataFrame):
        strategy = CramersVStrategy()
        with pytest.raises(InsufficientColumnsError) as excinfo:
            strategy.compute(numeric_dataframe)
        assert excinfo.value.found == []


class TestCorrelationRatioStrategy:
    def test_returns_symmetric_matrix(self, mixed_dataframe: pd.DataFrame):
        strategy = CorrelationRatioStrategy()
        result = strategy.compute(mixed_dataframe)
        np.testing.assert_array_almost_equal(result.values, result.values.T)

    def test_diagonal_is_one(self, mixed_dataframe: pd.DataFrame):
        strategy = CorrelationRatioStrategy()
        result = strategy.compute(mixed_dataframe)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_values_in_valid_range(self, mixed_dataframe: pd.DataFrame):
        strategy = CorrelationRatioStrategy()
        result = strategy.compute(mixed_dataframe)
        assert (result >= 0 - 1e-10).all().all()
        assert (result <= 1 + 1e-10).all().all()

    def test_detects_strong_association(self, mixed_dataframe: pd.DataFrame):
        strategy = CorrelationRatioStrategy()
        result = strategy.compute(mixed_dataframe)
        # category strongly determines value
        assert result.loc["category", "value"] > 0.8

    def test_numeric_only_frame_raises(self, numeric_dataframe: pd.DataFrame):
        strategy = CorrelationRatioStrategy()
        with pytest.raises(InsufficientColumnsError, match="1 categorical"):
            strategy.compute(numeric_dataframe)

    def test_categorical_only_frame_raises(self, categorical_dataframe: pd.DataFrame):
        strategy = CorrelationRatioStrategy()
        with pytest.raises(InsufficientColumnsError, match="1 numeric"):
            strategy.compute(categorical_dataframe)

    def test_constant_numeric_column_yields_zero(self):
        # zero total variance leaves nothing for the categories to explain
        df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "flat": [5.0] * 4})
        result = CorrelationRatioStrategy().compute(df)
        assert result.loc["cat", "flat"] == 0.0


class TestCorrelationRatioMissingValues:
    """Each pair must be measured on the rows where both columns are present.

    Taking the total sum of squares over every row while the between-group sum
    only covers rows with a known category divides two quantities measured on
    different populations, which understates the association.
    """

    def test_missing_category_does_not_understate_association(self):
        # 'cat' determines 'v' exactly on the rows where both are present
        df = pd.DataFrame(
            {
                "cat": ["a", "a", "b", "b", None, None],
                "v": [1.0, 1.0, 10.0, 10.0, 500.0, 600.0],
            }
        )
        result = CorrelationRatioStrategy().compute(df)
        assert result.loc["cat", "v"] == pytest.approx(1.0)

    def test_matches_the_same_frame_with_rows_dropped(self):
        df = pd.DataFrame(
            {
                "cat": ["a", "a", "b", "b", None, None],
                "v": [1.0, 2.0, 10.0, 11.0, 500.0, 600.0],
            }
        )
        with_nan = CorrelationRatioStrategy().compute(df).loc["cat", "v"]
        dropped = CorrelationRatioStrategy().compute(df.dropna()).loc["cat", "v"]
        assert with_nan == pytest.approx(dropped)

    def test_missing_numeric_values_are_excluded(self):
        df = pd.DataFrame(
            {
                "cat": ["a", "a", "b", "b"],
                "v": [1.0, np.nan, 10.0, 10.0],
            }
        )
        result = CorrelationRatioStrategy().compute(df)
        assert result.loc["cat", "v"] == pytest.approx(1.0)

    def test_no_overlapping_rows_raises(self):
        # the two columns are never present on the same row
        df = pd.DataFrame(
            {
                "cat": ["a", "b", None, None],
                "v": [np.nan, np.nan, 1.0, 2.0],
            }
        )
        with pytest.raises(NoCompleteObservationsError) as excinfo:
            CorrelationRatioStrategy().compute(df)
        assert set(excinfo.value.columns) == {"cat", "v"}

    def test_error_message_names_both_columns(self):
        df = pd.DataFrame({"cat": ["a", None], "v": [np.nan, 2.0]})
        with pytest.raises(NoCompleteObservationsError, match="'cat' and 'v'"):
            CorrelationRatioStrategy().compute(df)

    def test_single_surviving_group_yields_zero(self):
        # one group explains none of the spread within it
        df = pd.DataFrame(
            {
                "cat": ["a", "a", None, None],
                "v": [1.0, 9.0, 100.0, 200.0],
            }
        )
        result = CorrelationRatioStrategy().compute(df)
        assert result.loc["cat", "v"] == 0.0


class TestCramersVDegenerateInput:
    def test_single_valued_column_yields_zero(self):
        # a column with one category leaves no degrees of freedom
        df = pd.DataFrame({"varies": ["a", "b", "a", "b"], "constant": ["z"] * 4})
        result = CramersVStrategy().compute(df)
        assert result.loc["varies", "constant"] == 0.0
