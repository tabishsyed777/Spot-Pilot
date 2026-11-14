"""Exogenous workloads and market traces, independent of policy decisions."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Job:
    id: int
    arrival: int
    work: int
    deadline: int


@dataclass(frozen=True)
class Scenario:
    seed: int
    suite: str
    horizon: int
    budget: int
    jobs: tuple
    prices: tuple
    risks: tuple
    draws: tuple
    on_demand_price: int = 6
    checkpoint_storage_price: int = 1

    def to_dict(self):
        return asdict(self)


def generate(seed=7, suite='standard'):
    if suite not in ('standard', 'volatile', 'tight_budget'):
        raise ValueError('suite must be standard, volatile, or tight_budget')
    rng = np.random.default_rng(seed)
    horizon = 48
    budget = 150 if suite == 'tight_budget' else 220
    jobs = []
    for i in range(12):
        arrival = int(rng.integers(0, 21))
        work = int(rng.integers(3, 9))
        deadline = min(horizon, arrival + work + int(rng.integers(9, 24)))
        jobs.append(Job(i, arrival, work, deadline))
    prices, risks = [], []
    for t in range(horizon):
        if t % 8 == 0:
            price = int(rng.integers(1, 4 if suite != 'volatile' else 6))
            risk = float(rng.choice([.02, .06, .12, .2] if suite != 'volatile' else [.12,.25,.4]))
        prices.append(price); risks.append(risk)
    draws = tuple(tuple(float(v) for v in row) for row in rng.random((horizon, 2)))
    return Scenario(int(seed), suite, horizon, budget, tuple(jobs), tuple(prices), tuple(risks), draws)
