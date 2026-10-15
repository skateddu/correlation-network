"""Input validation utilities."""

from collections.abc import Collection

import numpy as np
import pandas as pd

from correlation_network.exceptions import NoCompleteObservationsError, NotFittedError

NOT_FITTED_MESSAGE = (
    "The instance is not fitted yet. "
    "Call 'fit' with appropriate arguments before using the estimator."
)


def check_is_fitted(
    estimator: object,
    attributes: set[str] | None = None,
    message: str = NOT_FITTED_MESSAGE,
) -> None:
    """Check whether an estimator has been fitted.

    Parameters
    ----------
    estimator : object
        The estimator instance to check.
    attributes : set[str] | None
        Fitted attributes to look for. Defaults to ``{"adjacency_matrix_"}``.
    message : str
        Error message raised when the estimator is not fitted.

    Raises
    ------
    NotFittedError
        If none of the expected fitted attributes are found.
    """
    if attributes is None:
        attributes = {"adjacency_matrix_"}
    object_attrs = set(vars(estimator))
    if not attributes.intersection(object_attrs):
        raise NotFittedError(message)


def complete_pair(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Restrict two series to the rows where both values are present.

    Parameters
    ----------
    x : pd.Series
        First column of the pair.
    y : pd.Series
        Second column of the pair, aligned with ``x``.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Both series, filtered to their pairwise-complete rows.

    Raises
    ------
    NoCompleteObservationsError
        If the two series share no rows where both values are present.
    """
    valid = x.notna() & y.notna()
    if not valid.any():
        raise NoCompleteObservationsError(columns=(str(x.name), str(y.name)))
    return x[valid], y[valid]


def check_pairwise_overlap(dataframe: pd.DataFrame) -> None:
    """Ensure every column pair shares at least one complete row.

    Matrix-level counterpart of :func:`complete_pair`, for strategies that
    delegate the pairwise computation to pandas, which reports such pairs as
    ``NaN`` rather than raising.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Columns to check.

    Raises
    ------
    NoCompleteObservationsError
        For the first column pair with no rows where both values are present.
    """
    present = dataframe.notna().to_numpy()
    if len(present) and present.all():
        return
    # Float matmul goes through BLAS; integer matmul does not and takes
    # seconds on large frames. Counts stay exact far beyond realistic sizes.
    present = present.astype(np.float64)
    overlap = present.T @ present
    rows, cols = np.nonzero(np.triu(overlap == 0, k=1))
    if rows.size:
        names = dataframe.columns
        raise NoCompleteObservationsError(
            columns=(str(names[rows[0]]), str(names[cols[0]]))
        )


def check_option(name: str, value: object, options: Collection[str]) -> None:
    """Ensure a string argument is one of the supported options.

    Parameters
    ----------
    name : str
        Argument name, used in the error message.
    value : object
        Value received.
    options : Collection[str]
        Accepted values.

    Raises
    ------
    ValueError
        If ``value`` is not in ``options``.
    """
    if value not in options:
        raise ValueError(
            f"Unknown {name} {value!r}. Available options: {tuple(options)}"
        )
