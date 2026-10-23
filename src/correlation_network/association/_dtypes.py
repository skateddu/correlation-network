"""Column type detection shared by the type-aware association strategies."""

from typing import Literal

import pandas as pd

# bool and pandas' "string" dtype hold labels, not quantities. select_dtypes
# leaves both out of "number", so unless they are listed here every
# type-aware strategy drops them without notice.
CATEGORICAL_DTYPES: tuple[Literal["object", "category", "bool", "string"], ...] = (
    "object",
    "category",
    "bool",
    "string",
)


def categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return the columns treated as categorical.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data.

    Returns
    -------
    list[str]
        Object, category, bool and string columns, in frame order.
    """
    return dataframe.select_dtypes(include=list(CATEGORICAL_DTYPES)).columns.tolist()


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return the columns treated as numeric (bool excluded).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data.

    Returns
    -------
    list[str]
        Numeric columns, in frame order.
    """
    return dataframe.select_dtypes(include="number").columns.tolist()
