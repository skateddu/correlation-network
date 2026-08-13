"""Tests for universal association strategies."""

import numpy as np
import pandas as pd
import pytest

from correlation_network.association._universal import (
    MutualInformationStrategy,
    PhiKStrategy,
)
from correlation_network.exceptions import InsufficientColumnsError


class TestMutualInformationStrategy:
    def test_returns_symmetric_matrix(self, mixed_dataframe: pd.DataFrame):
        result = MutualInformationStrategy().compute(mixed_dataframe)
        np.testing.assert_array_almost_equal(result.values, result.values.T)

    def test_diagonal_is_one(self, mixed_dataframe: pd.DataFrame):
        result = MutualInformationStrategy().compute(mixed_dataframe)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_values_in_valid_range(self, mixed_dataframe: pd.DataFrame):
        result = MutualInformationStrategy().compute(mixed_dataframe)
        assert (result >= -1e-10).all().all()
        assert (result <= 1 + 1e-10).all().all()

    def test_detects_strong_association(self, mixed_dataframe: pd.DataFrame):
        result = MutualInformationStrategy().compute(mixed_dataframe)
        # category determines value by construction; noise is independent
        assert result.loc["category", "value"] > result.loc["category", "noise"]

    def test_keeps_categorical_columns(self, mixed_dataframe: pd.DataFrame):
        result = MutualInformationStrategy().compute(mixed_dataframe)
        assert set(result.columns) == set(mixed_dataframe.columns)

    def test_bin_count_changes_the_result(self, numeric_dataframe: pd.DataFrame):
        coarse = MutualInformationStrategy(n_bins=2).compute(numeric_dataframe)
        fine = MutualInformationStrategy(n_bins=20).compute(numeric_dataframe)
        assert coarse.loc["x", "y"] != fine.loc["x", "y"]

    def test_fine_binning_inflates_independent_pairs(
        self, numeric_dataframe: pd.DataFrame
    ):
        # With few samples per bin, MI overestimates: x and z are independent by
        # construction, yet their score climbs steadily with the bin count.
        # This is a known bias, not a bug — it is why n_bins defaults to 10.
        scores = [
            MutualInformationStrategy(n_bins=b).compute(numeric_dataframe).loc["x", "z"]
            for b in (2, 10, 50)
        ]
        assert scores == sorted(scores)
        assert scores[0] < 0.05
        assert scores[-1] > 0.4

    def test_constant_column_yields_zero_association(self):
        df = pd.DataFrame({"varies": [1, 2, 3, 4, 5, 6], "constant": [7] * 6})
        result = MutualInformationStrategy(n_bins=3).compute(df)
        # zero entropy on one side means MI is undefined; treated as no association
        assert result.loc["varies", "constant"] == 0.0

    def test_single_column_raises(self, numeric_dataframe: pd.DataFrame):
        with pytest.raises(InsufficientColumnsError, match="at least 2 columns"):
            MutualInformationStrategy().compute(numeric_dataframe[["x"]])


class TestPhiKStrategy:
    def test_missing_dependency_raises_import_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "phik":
                raise ImportError("no phik")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="correlation-network\\[phik\\]"):
            PhiKStrategy().compute(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    def test_computes_matrix_when_available(self, mixed_dataframe: pd.DataFrame):
        pytest.importorskip("phik")
        result = PhiKStrategy().compute(mixed_dataframe)
        assert set(result.columns) == set(mixed_dataframe.columns)
        assert (result.values >= -1e-10).all()
