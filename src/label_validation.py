"""Validate a probability vector before persisting a teacher response."""
import math

ARCHETYPES = ("bee", "ant", "butterfly", "capybara", "leech")


def validate_distribution(data):
    values = {}
    for name in ARCHETYPES:
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a numeric probability")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between zero and one")
        values[name] = float(value)
    total = sum(values.values())
    if not 0.95 <= total <= 1.05:
        raise ValueError("probabilities must sum to approximately one")
    return {name: value / total for name, value in values.items()}
