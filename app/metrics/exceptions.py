"""Structured exceptions for the metric registry."""


class MetricError(Exception):
    """Base for all metric registry errors."""


class MetricNotFound(MetricError):
    """The requested metric_id or alias does not exist."""


class ScopeNotAllowed(MetricError):
    """The metric has no implementation for the current scope."""


class CapabilityNotSupported(MetricError):
    """The metric does not support the requested operation."""


class DimensionNotAllowed(MetricError):
    """The metric does not support this dimension in the current scope."""


class RegistryValidationError(MetricError):
    """Configuration or registry integrity check failed (startup fatal)."""
