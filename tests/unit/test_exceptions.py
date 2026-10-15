"""Tests for the exception hierarchy."""

import pytest

from correlation_network import (
    CorrelationNetworkError,
    InsufficientColumnsError,
    NoCompleteObservationsError,
    NotFittedError,
)

ALL_ERRORS = [
    NotFittedError("not fitted"),
    NoCompleteObservationsError(columns=("a", "b")),
    InsufficientColumnsError(method="pearson", required="2 numeric", found=[]),
]
ERROR_IDS = ["not_fitted", "no_complete_observations", "insufficient_columns"]


class TestExceptionHierarchy:
    @pytest.mark.parametrize("error", ALL_ERRORS, ids=ERROR_IDS)
    def test_every_error_is_catchable_through_the_base_class(self, error):
        with pytest.raises(CorrelationNetworkError):
            raise error

    @pytest.mark.parametrize("error", ALL_ERRORS, ids=ERROR_IDS)
    def test_every_error_remains_a_value_error(self, error):
        # existing `except ValueError` handlers must keep working
        assert isinstance(error, ValueError)

    def test_not_fitted_error_remains_an_attribute_error(self):
        assert isinstance(NotFittedError("not fitted"), AttributeError)

    def test_insufficient_columns_message_does_not_assume_fit_arguments(self):
        # strategies are public and can be used without CorrelationNetwork.fit
        error = InsufficientColumnsError(method="pearson", required="x", found=[])
        assert "columns=" not in str(error)
