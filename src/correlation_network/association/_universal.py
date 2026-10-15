"""Universal association strategies that work with any column type."""

from itertools import combinations
from typing import get_args

import numpy as np
import pandas as pd
from scipy.stats import entropy

from correlation_network._types import BinningMethod
from correlation_network._validation import (
    check_option,
    check_pairwise_overlap,
    complete_pair,
)
from correlation_network.association._dtypes import numeric_columns
from correlation_network.exceptions import InsufficientColumnsError


class PhiKStrategy:
    """PhiK correlation coefficient (requires optional phik dependency).

    Works with any combination of column types. Returns values in [0, 1].
    """

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute PhiK correlation matrix.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data with any column types.

        Returns
        -------
        pd.DataFrame
            Symmetric matrix of PhiK values in [0, 1].

        Raises
        ------
        ImportError
            If the ``phik`` package is not installed.
        NoCompleteObservationsError
            If a pair has no rows where both columns are present.
        """
        try:
            import phik  # noqa: F401
        except ImportError:
            raise ImportError(
                "PhiK strategy requires the 'phik' package. "
                "Install it with: pip install 'correlation-network[phik]'"
            ) from None

        # phik reports pairs without overlapping rows as NaN instead of raising
        check_pairwise_overlap(dataframe)
        return dataframe.phik_matrix()


class MutualInformationStrategy:
    """Normalized mutual information for any column type.

    Discretizes numeric columns before computing MI. Returns values in [0, 1].

    Parameters
    ----------
    n_bins : int
        Number of bins for numeric columns. Default ``10``.
    binning : BinningMethod
        ``"uniform"`` (default) splits the value range into equal-width bins;
        ``"quantile"`` uses equal-frequency bins, which a few extreme values
        cannot squeeze into a single bin. Duplicate quantile edges are merged,
        so heavily repeated values may yield fewer than ``n_bins`` bins.

    Raises
    ------
    ValueError
        If ``binning`` is not a supported method.
    """

    def __init__(self, n_bins: int = 10, binning: BinningMethod = "uniform") -> None:
        check_option(name="binning", value=binning, options=get_args(BinningMethod))
        self.n_bins = n_bins
        self.binning = binning

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized mutual information matrix.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data with any column types.

        Returns
        -------
        pd.DataFrame
            Symmetric matrix of normalized MI values in [0, 1].

        Raises
        ------
        InsufficientColumnsError
            If fewer than 2 columns are present.
        NoCompleteObservationsError
            If a pair has no rows where both columns are present.
        """
        if len(dataframe.columns) < 2:
            raise InsufficientColumnsError(
                method="mutual_information",
                required="at least 2 columns",
                found=dataframe.columns.tolist(),
            )
        discretized = self._discretize(dataframe)
        columns = discretized.columns.tolist()
        n = len(columns)
        matrix = np.eye(n)

        for i, j in combinations(range(n), 2):
            value = self._normalized_mi(
                x=discretized[columns[i]],
                y=discretized[columns[j]],
            )
            matrix[i, j] = value
            matrix[j, i] = value

        return pd.DataFrame(data=matrix, index=columns, columns=columns)

    def _discretize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Discretize numeric columns into bins."""
        result = dataframe.copy()
        numeric_cols = numeric_columns(result)
        result[numeric_cols] = result[numeric_cols].apply(self._bin)
        return result

    def _bin(self, column: pd.Series) -> pd.Series:
        """Map one numeric column to integer bin labels."""
        match self.binning:
            case "quantile":
                return pd.qcut(column, q=self.n_bins, labels=False, duplicates="drop")
            case _:
                return pd.cut(column, bins=self.n_bins, labels=False)

    @staticmethod
    def _normalized_mi(x: pd.Series, y: pd.Series) -> float:
        """Compute normalized mutual information between two discrete series."""
        x_clean, y_clean = complete_pair(x, y)

        contingency = pd.crosstab(index=x_clean, columns=y_clean)
        joint_prob = contingency.values / contingency.values.sum()

        marginal_x = joint_prob.sum(axis=1)
        marginal_y = joint_prob.sum(axis=0)

        h_x = entropy(marginal_x)
        h_y = entropy(marginal_y)

        if h_x == 0 or h_y == 0:
            return 0.0

        h_xy = entropy(joint_prob.ravel())
        mi = h_x + h_y - h_xy
        return mi / np.sqrt(h_x * h_y)
