"""Hierarchical Synaptic Network (HSN): the Python training side.

Python trains and runs experiments (gates G0-G4); Rust serves (``rust/``) and is exposed to
Python as the ``hsn_rs`` extension module. See ``HSN_Implementation_Plan.md`` at the repository
root for the phase and chunk plan.
"""

from importlib.metadata import version as _dist_version

__version__: str = _dist_version("hsn")

__all__ = ["__version__"]
