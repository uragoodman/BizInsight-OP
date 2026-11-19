"""Built-in whitelist analysis templates.

Importing this package loads every built-in template so its ``@register``
decorator populates the registry used by the answer node.
"""

# These imports are intentionally for registration side effects.
from app.analysis.templates import cohort, correlation, pareto, rfm, packages

__all__ = ["cohort", "correlation", "pareto", "rfm", "packages"]
