"""Main CorrelationNetwork class for building correlation/association graphs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from correlation_network._types import AssociationMethod

if TYPE_CHECKING:
    import polars as pl
from correlation_network._validation import check_is_fitted
from correlation_network.association._auto import AutoAssociationStrategy
from correlation_network.association._base import AssociationStrategy
from correlation_network.association._categorical import (
    CorrelationRatioStrategy,
    CramersVStrategy,
)
from correlation_network.association._correlation import (
    KendallStrategy,
    PearsonStrategy,
    SpearmanStrategy,
)
from correlation_network.association._universal import (
    MutualInformationStrategy,
    PhiKStrategy,
)

_STRATEGY_REGISTRY: dict[str, type] = {
    "pearson": PearsonStrategy,
    "spearman": SpearmanStrategy,
    "kendall": KendallStrategy,
    "cramers_v": CramersVStrategy,
    "correlation_ratio": CorrelationRatioStrategy,
    "phik": PhiKStrategy,
    "mutual_information": MutualInformationStrategy,
    "auto": AutoAssociationStrategy,
}

AVAILABLE_METHODS = tuple(_STRATEGY_REGISTRY.keys())

# Sentinel `method` for instances built by `from_matrix`: they carry no
# strategy, so this is deliberately not one of AVAILABLE_METHODS.
PRECOMPUTED_METHOD = "precomputed"

_DEFAULT_CARDINALITY_THRESHOLD = 0.95


def _is_consecutive_sequence(series: pd.Series) -> bool:
    """Check whether the distinct values form a run of consecutive integers."""
    values = np.sort(series.dropna().unique())
    return len(values) > 1 and bool((np.diff(values) == 1).all())


def _to_pandas(dataframe: Any) -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas if needed.

    Parameters
    ----------
    dataframe : pd.DataFrame | pl.DataFrame | pl.LazyFrame
        Input data.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame ready for processing.

    Raises
    ------
    TypeError
        If the input is not a supported DataFrame type.
    """
    if isinstance(dataframe, pd.DataFrame):
        return dataframe

    try:
        import polars as pl
    except ImportError:
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(dataframe).__name__}. "
            "Install polars for Polars support: "
            "pip install 'correlation-network[polars]'"
        ) from None

    if isinstance(dataframe, pl.LazyFrame):
        return dataframe.collect().to_pandas()
    if isinstance(dataframe, pl.DataFrame):
        return dataframe.to_pandas()

    raise TypeError(
        f"Expected a pandas or Polars DataFrame, got {type(dataframe).__name__}"
    )


class CorrelationNetwork:
    """Build a network graph from pairwise association measures.

    Parameters
    ----------
    method : AssociationMethod
        Association method to use. One of: ``"pearson"``, ``"spearman"``,
        ``"kendall"``, ``"cramers_v"``, ``"correlation_ratio"``,
        ``"phik"``, ``"mutual_information"``, ``"auto"``.
    threshold : float
        Minimum absolute association value to create an edge; the comparison
        is inclusive, so ``|value| == threshold`` keeps the edge.
        Must be in the range (0, 1).
    strategy : AssociationStrategy | None
        Custom strategy instance. Overrides ``method`` if provided.
    cardinality_threshold : float | None
        Maximum allowed cardinality ratio (nunique / nrows) for a column.
        String and categorical columns reaching this ratio are excluded as
        index-like variables; integer columns are excluded only if their
        values also form a consecutive sequence (row numbers, auto-increment
        keys). Set to ``None`` to disable the filter. Default ``0.95``.

    Attributes
    ----------
    association_matrix_ : pd.DataFrame
        Raw association matrix (available after ``fit``).
    adjacency_matrix_ : pd.DataFrame
        Thresholded association matrix (available after ``fit``).
    excluded_columns_ : list[str]
        Columns excluded during preprocessing (datetime or high-cardinality)
        or left out by the strategy because of an unsupported type.
    """

    def __init__(
        self,
        method: AssociationMethod = "pearson",
        threshold: float = 0.5,
        strategy: AssociationStrategy | None = None,
        cardinality_threshold: float | None = _DEFAULT_CARDINALITY_THRESHOLD,
    ) -> None:
        if method not in AVAILABLE_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Available options: {AVAILABLE_METHODS}"
            )
        if not (0 < threshold < 1):
            raise ValueError(
                f"The threshold must be in the range (0, 1), got {threshold}."
            )
        self.method = method
        self.threshold = threshold
        self.cardinality_threshold = cardinality_threshold
        self.strategy = strategy
        self._strategy = (
            strategy if strategy is not None else _STRATEGY_REGISTRY[method]()
        )

    def fit(
        self,
        dataframe: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> CorrelationNetwork:
        """Compute the association matrix and apply thresholding.

        Parameters
        ----------
        dataframe : pd.DataFrame | pl.DataFrame | pl.LazyFrame
            Input data. Polars DataFrames are converted to pandas
            internally (requires the ``polars`` extra).
        columns : list[str] | None
            Subset of columns to use. Uses all columns if None.

        Returns
        -------
        CorrelationNetwork
            The fitted instance (for method chaining).
        """
        dataframe = _to_pandas(dataframe)
        if columns is not None:
            dataframe = dataframe.loc[:, columns]

        dataframe = self._preprocess(dataframe)

        self.association_matrix_ = self._strategy.compute(dataframe)
        self._record_unused_columns(input_columns=dataframe.columns)
        self.adjacency_matrix_ = self._apply_threshold(self.association_matrix_)
        return self

    @classmethod
    def from_matrix(
        cls,
        matrix: pd.DataFrame | pl.DataFrame,
        *,
        threshold: float | None = None,
    ) -> CorrelationNetwork:
        """Create an instance from a pre-computed association matrix.

        The returned instance is immediately ready for ``transform()``
        without calling ``fit()``.

        Parameters
        ----------
        matrix : pd.DataFrame | pl.DataFrame
            Pre-computed symmetric association matrix.
            Index and columns must match. Polars DataFrames are
            converted to pandas internally.
        threshold : float | None
            If provided, values below this absolute threshold are zeroed
            in the adjacency matrix. If ``None``, the matrix is used as-is.

        Returns
        -------
        CorrelationNetwork
            Fitted instance ready for ``transform()``.

        Raises
        ------
        TypeError
            If ``matrix`` is not a supported DataFrame type.
        ValueError
            If the matrix is not square, labels don't match,
            or it is not symmetric.
        """
        is_pandas = isinstance(matrix, pd.DataFrame)
        matrix = _to_pandas(matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
        if not is_pandas:
            # Polars frames carry no row index, so rows can only be labelled
            # by position; the converted RangeIndex would never match.
            matrix.index = matrix.columns
        if not matrix.index.equals(matrix.columns):
            raise ValueError("Row index must match column index")
        if not np.allclose(matrix.values, matrix.values.T, atol=1e-8):
            raise ValueError("Matrix must be symmetric")

        if threshold is not None and not (0 < threshold < 1):
            raise ValueError(
                f"The threshold must be in the range (0, 1), got {threshold}."
            )

        instance = cls.__new__(cls)
        instance.method = PRECOMPUTED_METHOD
        instance.threshold = threshold
        instance.cardinality_threshold = None
        instance.strategy = None
        instance._strategy = None
        instance.excluded_columns_ = []
        instance.association_matrix_ = matrix
        # `_apply_threshold` is a no-op when `threshold` is None.
        instance.adjacency_matrix_ = instance._apply_threshold(matrix)

        return instance

    def transform(self) -> nx.Graph:
        """Build a NetworkX graph from the fitted adjacency matrix.

        Returns
        -------
        nx.Graph
            Weighted undirected graph with self-loops removed.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        check_is_fitted(self)
        graph = nx.from_pandas_adjacency(df=self.adjacency_matrix_)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph

    def fit_transform(
        self,
        dataframe: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> nx.Graph:
        """Fit and transform in a single step.

        Parameters
        ----------
        dataframe : pd.DataFrame | pl.DataFrame | pl.LazyFrame
            Input data. Polars DataFrames are converted to pandas
            internally (requires the ``polars`` extra).
        columns : list[str] | None
            Subset of columns to use.

        Returns
        -------
        nx.Graph
            Weighted undirected graph.
        """
        self.fit(dataframe, columns=columns)
        return self.transform()

    def _preprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Exclude datetime and high-cardinality columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data (already subsetted by ``columns`` if provided).

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with problematic columns removed.

        Raises
        ------
        ValueError
            If the DataFrame is empty or fewer than 2 columns remain.
        """
        if len(dataframe) == 0:
            raise ValueError("Cannot fit on an empty DataFrame.")

        excluded: list[str] = []

        # Exclude datetime columns
        datetime_cols = dataframe.select_dtypes(
            include=["datetime64", "datetimetz"]
        ).columns.tolist()
        if datetime_cols:
            warnings.warn(
                f"Datetime columns excluded: {datetime_cols}",
                UserWarning,
                stacklevel=3,
            )
            excluded.extend(datetime_cols)
            dataframe = dataframe.drop(columns=datetime_cols)

        if self.cardinality_threshold is not None:
            high_cardinality_cols = self._high_cardinality_columns(dataframe)
            if high_cardinality_cols:
                warnings.warn(
                    f"High-cardinality columns excluded: {high_cardinality_cols}",
                    UserWarning,
                    stacklevel=3,
                )
                excluded.extend(high_cardinality_cols)
                dataframe = dataframe.drop(columns=high_cardinality_cols)

        self.excluded_columns_ = excluded

        if len(dataframe.columns) < 2:
            raise ValueError(
                "Fewer than 2 columns remain after preprocessing. "
                f"Excluded columns: {excluded}"
            )

        return dataframe

    def _high_cardinality_columns(self, dataframe: pd.DataFrame) -> list[str]:
        """Return index-like columns whose cardinality ratio is too high.

        Continuous floats are never checked: they are naturally near-unique and
        still produce valid correlations. Integer columns must additionally be
        a consecutive sequence once sorted (row numbers, auto-increment keys);
        on small frames genuine integer measurements are near-unique as well.
        """
        candidates = dataframe.select_dtypes(include=["integer", "object", "category"])
        ratio = candidates.nunique() / len(dataframe)
        flagged = ratio[ratio >= self.cardinality_threshold].index
        return [
            col
            for col in flagged
            if not pd.api.types.is_integer_dtype(dataframe[col])
            or _is_consecutive_sequence(dataframe[col])
        ]

    def _record_unused_columns(self, input_columns: pd.Index) -> None:
        """Warn about and record columns the strategy left out of its matrix.

        Strategies only use the column types they support (e.g. correlations
        skip strings); without this, such columns would vanish from the graph
        without notice.
        """
        computed = set(self.association_matrix_.columns)
        unused = [col for col in input_columns if col not in computed]
        if unused:
            warnings.warn(
                f"Columns not used by the strategy (unsupported type): {unused}",
                UserWarning,
                stacklevel=3,
            )
            self.excluded_columns_.extend(unused)

    def _apply_threshold(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Zero out values below the threshold (by absolute value).

        A ``None`` threshold means "keep everything", which is the state an
        instance built by :meth:`from_matrix` without a threshold is left in.
        The comparison is inclusive, matching the documented contract that
        ``threshold`` is the *minimum* value that still creates an edge.
        """
        if self.threshold is None:
            return matrix
        condition = (matrix >= self.threshold) | (matrix <= -self.threshold)
        return matrix.where(cond=condition, other=0)

    def __repr__(self) -> str:
        if self.strategy is not None:
            source = f"strategy={type(self.strategy).__name__}"
        else:
            source = f"method={self.method!r}"
        return f"CorrelationNetwork({source}, threshold={self.threshold})"
