"""Exception hierarchy exposed by the AutoML package."""


class AutoMLError(Exception):
    """Base exception for the public AutoML API."""


class ConfigError(AutoMLError):
    """Raised when an AutoML configuration is inconsistent."""


class SchemaError(AutoMLError):
    """Raised when an input dataset violates the fitted schema."""


class NotFittedError(AutoMLError):
    """Raised when an operation requires a trained task."""


class MissingDependencyError(AutoMLError):
    """Raised when the selected engine is not installed."""


class UnsupportedBackendError(ConfigError):
    """Raised when a backend/engine pair is unsupported."""


class ArtifactError(AutoMLError):
    """Base exception for model artifact failures."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when an artifact is incomplete or corrupted."""


class RemoteExecutionError(AutoMLError):
    """Raised when remote submission or completion cannot produce a valid result."""
