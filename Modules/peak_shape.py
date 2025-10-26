"""Utilities for building chromatographic peak-shape descriptors and
computing similarity scores between LC-MS features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "PeakShapeVectors",
    "build_peak_shape_vectors",
    "peak_shape_similarity",
]


@dataclass
class PeakShapeVectors:
    """Container for per-feature peak-shape descriptors."""

    vectors: pd.DataFrame

    def similarity(self, feature_a: str, feature_b: str) -> Optional[float]:
        """Return the cosine similarity between two feature descriptors."""
        if self.vectors is None or self.vectors.empty:
            return None
        try:
            vec_a = self.vectors.loc[feature_a]
            vec_b = self.vectors.loc[feature_b]
        except KeyError:
            return None

        mask = ~(vec_a.isna() | vec_b.isna())
        if not mask.any():
            return None

        a = vec_a[mask].to_numpy(dtype=float)
        b = vec_b[mask].to_numpy(dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return None
        # Cosine similarity in [-1, 1]
        score = float(np.dot(a, b) / (norm_a * norm_b))
        # Numerical noise might produce a value slightly outside the bounds
        return max(min(score, 1.0), -1.0)


def _weighted_average(values: pd.Series, weights: pd.Series) -> Optional[float]:
    mask = (~values.isna()) & (~weights.isna()) & (weights > 0)
    if not mask.any():
        return None
    v = values[mask].to_numpy(dtype=float)
    w = weights[mask].to_numpy(dtype=float)
    total = w.sum()
    if total <= 0:
        return float(v.mean())
    return float(np.dot(v, w) / total)


def build_peak_shape_vectors(feature_df: pd.DataFrame) -> Optional[PeakShapeVectors]:
    """Compute chromatographic peak-shape descriptors for each feature.

    The vectors include log peak width, log m/z span, and the apex position
    within the peak window. Each descriptor is averaged across samples using
    peak area as weight. When mandatory columns are missing, ``None`` is
    returned so that the caller can gracefully fall back to the legacy
    behaviour without peak-shape filtering.
    """

    required_columns = {
        "feature",
        "rt",
        "area",
        "peak_rt_start",
        "peak_rt_end",
        "peak_mz_min",
        "peak_mz_max",
    }

    if not required_columns.issubset(feature_df.columns):
        return None

    df = feature_df[list(required_columns)].copy(deep=True)
    numeric_cols = required_columns - {"feature"}
    df.loc[:, list(numeric_cols)] = df.loc[:, list(numeric_cols)].apply(
        pd.to_numeric, errors="coerce"
    )

    df["peak_width"] = (df["peak_rt_end"] - df["peak_rt_start"]).clip(lower=0)
    df["log_peak_width"] = np.log1p(df["peak_width"])
    df["mz_span"] = (df["peak_mz_max"] - df["peak_mz_min"]).clip(lower=0)
    df["log_mz_span"] = np.log1p(df["mz_span"])

    with np.errstate(divide="ignore", invalid="ignore"):
        df["apex_fraction"] = np.where(
            df["peak_width"] > 0,
            (df["rt"] - df["peak_rt_start"]) / df["peak_width"],
            np.nan,
        )
    df["apex_fraction"] = df["apex_fraction"].clip(lower=0, upper=1)

    df["area_weight"] = df["area"].where(df["area"] > 0)

    aggregations = {
        "shape_log_width": ("log_peak_width", _weighted_average),
        "shape_log_mz_span": ("log_mz_span", _weighted_average),
        "shape_apex_fraction": ("apex_fraction", _weighted_average),
    }

    records = {}
    for feature, group in df.groupby("feature"):
        weights = group["area_weight"]
        row = {}
        for column_name, (source_column, reducer) in aggregations.items():
            value = reducer(group[source_column], weights)
            row[column_name] = value if value is not None else np.nan
        records[feature] = row

    if not records:
        return None

    vectors = pd.DataFrame.from_dict(records, orient="index").sort_index()
    # Normalise descriptors column-wise to zero mean / unit variance when possible
    for column in vectors.columns:
        series = vectors[column]
        if series.notna().sum() < 2:
            continue
        centred = series - series.mean()
        std = centred.std(ddof=0)
        if std > 0:
            vectors[column] = centred / std
        else:
            vectors[column] = centred

    return PeakShapeVectors(vectors=vectors)


def peak_shape_similarity(
    vectors: Optional[PeakShapeVectors], feature_a: str, feature_b: str
) -> Optional[float]:
    """Convenience wrapper to compute similarity between two features."""

    if vectors is None:
        return None
    return vectors.similarity(feature_a, feature_b)
