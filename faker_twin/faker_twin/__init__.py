"""faker_twin: create fake twins of your data files with the same structure."""

from .generator import TableFaker
from .profiler import profile_table

__all__ = ["TableFaker", "profile_table"]
__version__ = "0.1.0"
