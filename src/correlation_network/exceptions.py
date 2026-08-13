"""Custom exception hierarchy for correlation-network."""


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before fitting.

    Inherits from both ValueError and AttributeError for compatibility
    with scikit-learn-style exception handling.
    """


class NoCompleteObservationsError(ValueError):
    """Raised when a column pair shares no rows where both values are present.

    Attributes
    ----------
    columns : tuple[str, str]
        The pair of column names with no overlapping complete rows.
    """

    def __init__(self, columns: tuple[str, str]) -> None:
        self.columns = columns
        left, right = columns
        super().__init__(
            f"Columns {left!r} and {right!r} share no rows where both values "
            f"are present, so their association is undefined. Drop or impute "
            f"the missing values, or exclude one of the columns."
        )


class InsufficientColumnsError(ValueError):
    """Raised when a strategy receives too few columns of the type it needs.

    Attributes
    ----------
    method : str
        Name of the strategy that rejected the input.
    required : str
        Human-readable description of the columns the strategy needs.
    found : list[str]
        Columns of the required type that were actually present.
    """

    def __init__(self, method: str, required: str, found: list[str]) -> None:
        self.method = method
        self.required = required
        self.found = found
        super().__init__(
            f"{method} requires {required}, but the DataFrame provides "
            f"{len(found)}: {found}. Pick a method suited to these column "
            f"types, or pass `columns=` to select compatible ones."
        )
