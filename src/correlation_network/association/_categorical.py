"""Association strategies for categorical and mixed-type pairs."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from correlation_network.exceptions import (
    InsufficientColumnsError,
    NoCompleteObservationsError,
)


class CramersVStrategy:
    """Cramer's V association measure for categorical-categorical pairs.

    Computes the bias-corrected Cramer's V for all pairs of categorical
    columns, returning values in [0, 1].
    """

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute Cramer's V matrix for categorical columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data. Only object/category dtype columns are used.

        Returns
        -------
        pd.DataFrame
            Symmetric matrix of Cramer's V values in [0, 1].

        Raises
        ------
        InsufficientColumnsError
            If fewer than 2 categorical columns are present.
        """
        categorical_cols = dataframe.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if len(categorical_cols) < 2:
            raise InsufficientColumnsError(
                method="cramers_v",
                required="at least 2 categorical columns",
                found=categorical_cols,
            )
        n_cols = len(categorical_cols)
        matrix = np.eye(n_cols)

        for i, j in combinations(range(n_cols), 2):
            value = self._cramers_v(
                x=dataframe[categorical_cols[i]],
                y=dataframe[categorical_cols[j]],
            )
            matrix[i, j] = value
            matrix[j, i] = value

        return pd.DataFrame(
            data=matrix, index=categorical_cols, columns=categorical_cols
        )

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Compute bias-corrected Cramer's V between two categorical series."""
        contingency = pd.crosstab(index=x, columns=y)
        chi2 = stats.chi2_contingency(observed=contingency)[0]
        n = contingency.values.sum()
        r, k = contingency.shape

        # Bias correction
        phi2 = chi2 / n
        phi2_corrected = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corrected = r - ((r - 1) ** 2) / (n - 1)
        k_corrected = k - ((k - 1) ** 2) / (n - 1)
        denominator = min(k_corrected - 1, r_corrected - 1)

        if denominator <= 0:
            return 0.0
        return np.sqrt(phi2_corrected / denominator)


class CorrelationRatioStrategy:
    """Correlation ratio (eta) for categorical-numeric pairs.

    For each pair of (categorical, numeric) columns, computes the
    correlation ratio in [0, 1].
    """

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation ratio matrix for mixed-type columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data with both categorical and numeric columns.

        Each pair is evaluated on the rows where both columns are present,
        matching the pairwise-complete convention the other strategies use.

        Returns
        -------
        pd.DataFrame
            Symmetric matrix of correlation ratio values in [0, 1].

        Raises
        ------
        InsufficientColumnsError
            If either no categorical or no numeric column is present.
        NoCompleteObservationsError
            If a pair has no rows where both columns are present.
        """
        cat_cols = dataframe.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        num_cols = dataframe.select_dtypes(include="number").columns.tolist()
        if not cat_cols or not num_cols:
            raise InsufficientColumnsError(
                method="correlation_ratio",
                required="at least 1 categorical and 1 numeric column",
                found=cat_cols + num_cols,
            )
        all_cols = cat_cols + num_cols
        n = len(all_cols)
        matrix = np.zeros((n, n))
        col_index = {col: i for i, col in enumerate(all_cols)}

        # Per-column stats cannot be pre-computed: each pair is restricted to
        # its own complete rows, so the grand mean and total sum of squares
        # depend on which categorical column it is paired with.
        for cat_col in cat_cols:
            for num_col in num_cols:
                value = self._correlation_ratio(
                    categories=dataframe[cat_col],
                    values=dataframe[num_col],
                )
                ci = col_index[cat_col]
                ni = col_index[num_col]
                matrix[ci, ni] = value
                matrix[ni, ci] = value

        np.fill_diagonal(matrix, 1.0)

        return pd.DataFrame(data=matrix, index=all_cols, columns=all_cols)

    @staticmethod
    def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
        """Compute correlation ratio (eta) for a categorical-numeric pair.

        Restricted to rows where both columns are present. Computing the total
        sum of squares over every row while the between-group sum only covers
        rows with a known category would divide two quantities measured on
        different populations, understating the association.

        Raises
        ------
        NoCompleteObservationsError
            If the two columns share no rows where both values are present.
        """
        valid = categories.notna() & values.notna()
        if not valid.any():
            raise NoCompleteObservationsError(
                columns=(str(categories.name), str(values.name))
            )

        categories = categories[valid]
        values = values[valid]

        grand_mean = values.mean()
        ss_total = ((values - grand_mean) ** 2).sum()
        # A constant column has no variance for the categories to explain.
        # That is uninformative rather than undefined, so report no association.
        if ss_total == 0:
            return 0.0

        group_stats = values.groupby(categories, observed=True).agg(["mean", "count"])
        ss_between = (
            group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2
        ).sum()
        return np.sqrt(ss_between / ss_total)
