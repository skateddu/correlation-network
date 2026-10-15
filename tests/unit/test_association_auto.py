"""Tests for the auto-detection association strategy."""

import pandas as pd
import pytest

from correlation_network.association._auto import AutoAssociationStrategy
from correlation_network.exceptions import NoCompleteObservationsError


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


class TestAutoMissingValues:
    def test_numeric_pair_without_overlap_raises(self, disjoint_numeric_dataframe: pd.DataFrame):
        df = disjoint_numeric_dataframe.assign(cat=list("ababab"))
        with pytest.raises(NoCompleteObservationsError):
            AutoAssociationStrategy().compute(df)


class TestAutoColumnTypes:
    @pytest.fixture()
    def frame_with_flag(self) -> pd.DataFrame:
        """'flag' fully determines 'value'; 'other' is unrelated noise."""
        flag = [True, False] * 50
        return pd.DataFrame(
            {
                "flag": flag,
                "value": [10.0 if f else 0.0 for f in flag],
                "other": [float(i % 7) for i in range(100)],
            }
        )

    def test_bool_column_is_kept(self, frame_with_flag: pd.DataFrame):
        # select_dtypes("number") excludes bool, which used to drop it silently
        result = AutoAssociationStrategy().compute(frame_with_flag)
        assert "flag" in result.columns

    def test_bool_column_is_treated_as_categorical(self, frame_with_flag: pd.DataFrame):
        result = AutoAssociationStrategy().compute(frame_with_flag)
        assert result.loc["flag", "value"] == pytest.approx(1.0)

    def test_string_dtype_column_is_treated_as_categorical(self):
        df = pd.DataFrame(
            {
                "label": pd.array(["a", "b"] * 20, dtype="string"),
                "value": [1.0, 5.0] * 20,
                "other": [float(i % 3) for i in range(40)],
            }
        )
        result = AutoAssociationStrategy().compute(df)
        assert result.loc["label", "value"] == pytest.approx(1.0)
