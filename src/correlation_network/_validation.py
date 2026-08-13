"""Input validation utilities."""

from correlation_network.exceptions import NotFittedError

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
