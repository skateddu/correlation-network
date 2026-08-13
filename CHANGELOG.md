# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
