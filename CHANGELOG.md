# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CorrelationNetworkError`, a common base class for every library error;
  each error keeps its previous built-in base (`ValueError`, `AttributeError`)
- `MutualInformationStrategy(binning="quantile")` for equal-frequency bins,
  robust to outliers that squeeze equal-width bins (default stays `"uniform"`)

### Changed

- `AutoAssociationStrategy` parameters are typed as `AssociationStrategy` and
  default only when `None` is passed; `NetworkVisualizer.save()` accepts
  `os.PathLike` paths
- `InsufficientColumnsError` message no longer suggests `columns=`, which only
  exists on `CorrelationNetwork.fit`

### Fixed

- Kamada-Kawai layout (the default) now converts association strength to
  distance (`1 - |weight|`); it previously drew the strongest associations as
  the longest edges
- `CentralityAnalyzer.eigenvector()` uses absolute weights, so signed graphs
  no longer produce negative centrality scores
- Cramer's V no longer applies Yates' continuity correction on 2x2 tables,
  which understated the association (a perfect 2x2 association now scores 1)
- Pearson, Spearman, Kendall, Cramer's V, Mutual Information and PhiK now
  raise `NoCompleteObservationsError` for a column pair with no complete rows,
  as documented, instead of returning `NaN` (silently dropped as "no edge"),
  a raw SciPy `ValueError`, or a fake `0.0`. This also applies to columns that
  are entirely missing
- `CorrelationNetwork.from_matrix()` accepts Polars matrices; rows are
  labelled by column order since Polars has no row index
- Bool and pandas `string` columns are treated as categorical by
  `method="auto"`, Cramer's V and the correlation ratio; `auto` used to drop
  them silently
- Columns a strategy cannot use (e.g. strings with `pearson`) now raise a
  `UserWarning` and are recorded in `excluded_columns_` instead of vanishing
  from the graph without notice
- The high-cardinality filter no longer excludes genuine integer features on
  small frames: integer columns are dropped only when their values form a
  consecutive sequence
- `repr()` names the custom strategy instead of the unused default method
- Edge colormap keeps 0 at its centre for single-signed weights, so weak
  positive associations are no longer painted in the "negative" colour
- `NetworkVisualizer.plot()` raises `ValueError` for an unknown `layout` or
  `centrality_metric` (previously `KeyError`, or any analyzer attribute was
  accepted as a metric)

## [1.0.0] - 2026-08-13

Initial release.

### Added

#### Core API

- `CorrelationNetwork` with a scikit-learn-style `fit` / `transform` /
  `fit_transform` pipeline
- `from_matrix()` class method for pre-computed association matrices
- Strategy pattern for association measures via the `AssociationStrategy`
  protocol; any object implementing `compute(df) -> pd.DataFrame` can be passed
  through the `strategy` parameter
- Polars DataFrame and LazyFrame input, converted internally (via the `polars`
  extra)
- Inline type information (PEP 561, `py.typed`)

#### Association measures

- Correlation strategies: Pearson, Spearman, Kendall
- Categorical strategies: Cramer's V (bias-corrected), Correlation Ratio (eta)
- Universal strategies: Mutual Information (normalized), PhiK (via the `phik`
  extra)
- `AutoAssociationStrategy`, dispatching by column type: Spearman for
  numeric pairs, Cramer's V for categorical pairs, eta for mixed pairs
- Pairs are evaluated on their pairwise-complete rows, so missing values in one
  column do not distort an unrelated pair
- Inclusive thresholding: an association equal to `threshold` creates an edge

#### Analysis and rendering

- `CentralityAnalyzer` with degree, strength, betweenness, eigenvector and
  closeness, individually or as a combined `summary()` table
- Path-based metrics convert association strength to distance
  (`distance = 1 - |weight|`), so shortest paths follow strongly associated
  pairs; `strength()` keeps raw magnitudes
- `NetworkVisualizer` with 5 layouts (kamada_kawai, spring, circular, shell,
  spectral), a continuous edge colormap with colorbar, a two-colour
  positive/negative fallback, community colouring via greedy modularity,
  centrality-scaled node sizes, per-edge alpha, and a `save()` helper

#### Validation

- Datetime and high-cardinality column exclusion during preprocessing, recorded
  in `excluded_columns_` and reported through warnings
- `NotFittedError` with a `check_is_fitted` guard
- `InsufficientColumnsError` when a method receives no columns of the type it
  needs, carrying the `method`, the `required` types, and the columns `found`
- `NoCompleteObservationsError` when a column pair shares no rows where both
  values are present, carrying the offending `columns`

#### Project

- Unit and integration test suites (139 tests, 99% coverage)
- GitHub Actions CI: lint, format, and a test matrix over Python 3.10–3.13,
  gated at 80% coverage

[1.0.0]: https://github.com/skateddu/correlation-network/releases/tag/v1.0.0
