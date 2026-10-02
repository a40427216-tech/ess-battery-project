"""Importable target transform so persisted models reload outside the CLI."""
import numpy as np


def power10(value):
    return np.power(10.0, value)
