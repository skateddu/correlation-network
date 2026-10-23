"""Correlation-based association strategies for numeric-numeric pairs."""

import pandas as pd

from correlation_network._types import CorrelationMethod
from correlation_network._validation import check_pairwise_overlap
from correlation_network.association._dtypes import numeric_columns
from correlation_network.exceptions import InsufficientColumnsError


class _CorrelationStrategy:
    """Base class for pandas correlation methods."""

    _method: CorrelationMethod

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix using pandas.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data with numeric columns.

        Returns
        -------
        pd.DataFrame
            Symmetric correlation matrix with values in [-1, 1].

        Raises
        ------
        InsufficientColumnsError
            If fewer than 2 numeric columns are present.
        NoCompleteObservationsError
            If a pair has no rows where both columns are present.
        """
        numeric_cols = numeric_columns(dataframe)
        if len(numeric_cols) < 2:
            raise InsufficientColumnsError(
                method=self._method,
                required="at least 2 numeric columns",
                found=numeric_cols,
            )
        matrix = dataframe.corr(method=self._method, numeric_only=True)
        # numeric_only also keeps bool columns, so check the columns pandas
        # actually used rather than numeric_cols
        check_pairwise_overlap(dataframe[matrix.columns])
        return matrix


class PearsonStrategy(_CorrelationStrategy):
    """Pearson linear correlation coefficient."""

    _method: CorrelationMethod = "pearson"


class SpearmanStrategy(_CorrelationStrategy):
    """Spearman rank correlation coefficient."""

    _method: CorrelationMethod = "spearman"


class KendallStrategy(_CorrelationStrategy):
    """Kendall tau rank correlation coefficient."""

    _method: CorrelationMethod = "kendall"
