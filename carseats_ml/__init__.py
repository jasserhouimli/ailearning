"""Reusable code for the Carseats salary/price classification study.

The package turns the dataset into a binary ``High`` sales label, prepares the
scaled features, and provides model factories plus evaluation helpers.
"""

from __future__ import annotations

from carseats_ml import data, evaluate, models

__all__ = ["data", "evaluate", "models"]
