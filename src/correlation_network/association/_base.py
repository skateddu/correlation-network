"""Base protocol for association strategies."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class AssociationStrategy(Protocol):
    """Protocol that all association strategies must implement."""

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute a symmetric association matrix from a DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data with columns to compute associations for.

        Returns
        -------
        pd.DataFrame
            Symmetric matrix of association values.
        """
        ...
