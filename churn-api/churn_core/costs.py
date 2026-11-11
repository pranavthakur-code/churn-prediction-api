"""Business cost model used to choose the decision threshold.

Per customer we decide between two actions:

    do nothing : if the customer is a churner we lose their value (L)
    make an offer: we always pay the offer cost (O); a churner who gets an
                   offer is saved with probability s, otherwise still lost.

Expected cost of offering  = O + p * (1 - s) * L
Expected cost of ignoring  =     p * L
Offering is cheaper when   p > O / (s * L)      <- analytic threshold

The analytic threshold assumes calibrated probabilities, so training also scans
thresholds on out-of-fold predictions and reports both.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    customer_value: float = 300.0   # L: margin lost when a customer churns (USD)
    offer_cost: float = 30.0        # O: cost of one retention offer (USD)
    save_rate: float = 0.5          # s: share of churners an offer retains

    def __post_init__(self):
        if self.customer_value <= 0 or self.offer_cost < 0:
            raise ValueError("customer_value must be > 0 and offer_cost >= 0")
        if not 0 < self.save_rate <= 1:
            raise ValueError("save_rate must be in (0, 1]")

    def analytic_threshold(self) -> float:
        return min(1.0, self.offer_cost / (self.save_rate * self.customer_value))

    def total_cost(self, y_true, p, threshold: float) -> float:
        """Expected total cost in USD of acting on `p >= threshold`."""
        y = np.asarray(y_true, dtype=float)
        offer = np.asarray(p) >= threshold
        cost_offer = self.offer_cost + y * (1 - self.save_rate) * self.customer_value
        cost_ignore = y * self.customer_value
        return float(np.where(offer, cost_offer, cost_ignore).sum())

    def baselines(self, y_true) -> dict:
        n = len(y_true)
        return {
            "no_action": self.total_cost(y_true, np.zeros(n), 1.01),
            "offer_to_everyone": self.total_cost(y_true, np.ones(n), 0.0),
        }

    def best_threshold(self, y_true, p, grid=None):
        """Threshold with the lowest expected cost. Ties go to the higher one
        (fewer offers). Returns (threshold, cost)."""
        grid = np.round(np.arange(0.02, 0.98, 0.01), 2) if grid is None else grid
        costs = np.array([self.total_cost(y_true, p, t) for t in grid])
        i = len(costs) - 1 - int(np.argmin(costs[::-1]))
        return float(grid[i]), float(costs[i])

    def as_dict(self) -> dict:
        return {**asdict(self), "analytic_threshold": self.analytic_threshold()}
