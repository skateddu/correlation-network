"""Auto-detection strategy that dispatches by column type."""

import numpy as np
import pandas as pd

from correlation_network.association._categorical import (
    CorrelationRatioStrategy,
    CramersVStrategy,
)
from correlation_network.association._correlation import SpearmanStrategy


class AutoAssociationStrategy:
    """Automatically selects the appropriate association measure per column pair.

    Default dispatch:
    - numeric-numeric: Spearman correlation
    - categorical-categorical: Cramer's V
    - categorical-numeric: Correlation ratio (eta)

    Notes
    -----
    Numeric-numeric correlations are reduced to their **absolute value**, so
    the resulting matrix is in [0, 1] and the sign of a negative correlation
    is lost. This keeps the three measures on a comparable scale (Cramer's V
    and eta are already non-negative), but means ``method="auto"`` never
    produces negative edge weights — the positive/negative edge colouring in
    :class:`~correlation_network.NetworkVisualizer` has no effect on such a
    graph. Use an explicit correlation method to retain the sign.

    Parameters
    ----------
    numeric_strategy : object | None
        Strategy for numeric-numeric pairs. Defaults to SpearmanStrategy.
    categorical_strategy : object | None
        Strategy for categorical-categorical pairs. Defaults to CramersVStrategy.
    mixed_strategy : object | None
        Strategy for categorical-numeric pairs. Defaults to CorrelationRatioStrategy.
    """

    def __init__(
        self,
        numeric_strategy: object | None = None,
        categorical_strategy: object | None = None,
        mixed_strategy: object | None = None,
    ) -> None:
        self._numeric = numeric_strategy or SpearmanStrategy()
        self._categorical = categorical_strategy or CramersVStrategy()
        self._mixed = mixed_strategy or CorrelationRatioStrategy()

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute a combined association matrix using type-aware dispatch.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data with numeric and/or categorical columns.

        Returns
        -------
        pd.DataFrame
            Symmetric association matrix combining all column types.
        """
        num_cols = dataframe.select_dtypes(include="number").columns.tolist()
        cat_cols = dataframe.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        all_cols = num_cols + cat_cols
        n = len(all_cols)
        matrix = np.eye(n)
        col_index = {col: i for i, col in enumerate(all_cols)}

        # Numeric-numeric block: assign via numpy fancy indexing
        if len(num_cols) > 1:
            num_matrix = self._numeric.compute(dataframe[num_cols])
            num_idx = np.array([col_index[c] for c in num_cols])
            matrix[np.ix_(num_idx, num_idx)] = num_matrix.abs().values

        # Categorical-categorical block: assign via numpy fancy indexing
        if len(cat_cols) > 1:
            cat_matrix = self._categorical.compute(dataframe[cat_cols])
            cat_idx = np.array([col_index[c] for c in cat_cols])
            matrix[np.ix_(cat_idx, cat_idx)] = cat_matrix.values

        # Mixed block (cat-num): assign via numpy fancy indexing
        if num_cols and cat_cols:
            mixed_matrix = self._mixed.compute(dataframe[num_cols + cat_cols])
            cat_idx = np.array([col_index[c] for c in cat_cols])
            num_idx = np.array([col_index[c] for c in num_cols])
            sub = mixed_matrix.loc[cat_cols, num_cols].values
            matrix[np.ix_(cat_idx, num_idx)] = sub
            matrix[np.ix_(num_idx, cat_idx)] = sub.T

        # Restore diagonal after block assignments
        np.fill_diagonal(matrix, 1.0)

        return pd.DataFrame(data=matrix, index=all_cols, columns=all_cols)
