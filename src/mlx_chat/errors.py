"""Application errors."""


class MlxChatError(Exception):
    """An expected error that should be shown without a traceback."""


class ServerUnavailableError(MlxChatError):
    """The inference server is unavailable."""


class ApiError(MlxChatError):
    """The inference API rejected a request or returned malformed data."""
