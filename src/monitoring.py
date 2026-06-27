"""Model monitoring utilities (population stability / drift)."""

from __future__ import annotations

import numpy as np


def population_stability_index(expected, actual, bins: int = 10) -> float:
    """Population Stability Index (PSI) between a reference and a current sample.

    Bins are quantile edges learned from ``expected`` (the reference), so each
    reference bin holds ~1/bins of the mass. PSI then sums, over the bins,
    ``(actual%% - expected%%) * ln(actual%% / expected%%)``.

    Rule of thumb: PSI < 0.10 estável, 0.10–0.25 mudança moderada,
    > 0.25 mudança significativa (investigar/retreinar).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Quantile edges from the reference; open tails so out-of-range values count.
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)  # guard against duplicate edges (ties)
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    eps = 1e-6
    exp_pct = np.clip(exp_counts / exp_counts.sum(), eps, None)
    act_pct = np.clip(act_counts / act_counts.sum(), eps, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
