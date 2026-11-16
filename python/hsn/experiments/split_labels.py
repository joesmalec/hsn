"""Shared query-slot schema for evaluation and torch-free experiment reports.

synthetic_v1 has at most four delayed queries; synthetic_v2 has up to eight.
This evaluator limit is separate from the frozen generator's default parameters.
"""

MAX_QUERIES = 8
SLOT_NAMES: tuple[str, ...] = ("first", *(f"later/{q}" for q in range(1, MAX_QUERIES)))
LATER_SLOTS: tuple[str, ...] = SLOT_NAMES[1:]
OTHER = "other"
NAMES: tuple[str, ...] = (*SLOT_NAMES, OTHER)
