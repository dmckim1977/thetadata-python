"""Module that contains custom exceptions."""
import enum
from typing import Any, Optional


class ThetadataError(Exception):
    """Base exception class for all Thetadata exceptions."""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ResponseParseError(ThetadataError):
    """Raised if the API failed to parse a Terminal response."""


class EnumParseError(ThetadataError):
    """Raised when a value cannot be parsed into an associated enum member."""
    def __init__(self, value: Any, enm: Any):
        """Create a new EnumParseError

        :param value: The value that cannot be parsed.
        :param enm: The enum.
        """
        assert issubclass(enm, enum.Enum), "Cannot create an EnumParseError with a non-enum."
        details = {
            "value": value,
            "enum_name": enm.__name__
        }
        super().__init__(f"Value {value} cannot be parsed into a {enm.__name__}!", details)


class ResponseError(ThetadataError):
    """Raised if there is an error in the body of a response."""


class NoDataError(ThetadataError):
    """Raised if no data is available for this request."""


class ConnectionError(ThetadataError):
    """Raised if connection to the server fails."""
    def __init__(self, message: str, host: Optional[str] = None, port: Optional[int] = None):
        details = {"host": host, "port": port} if host and port else {}
        super().__init__(f"Connection error: {message}", details)


class ThetadataValidationError(ThetadataError):
    """Exception raised when request parameters fail validation."""
    @classmethod
    def from_pydantic(cls, pydantic_error) -> 'ThetadataValidationError':
        """Convert a Pydantic ValidationError to our custom ValidationError."""
        details = {}
        for error in pydantic_error.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            details[field] = error["msg"]
        return cls("Validation failed", details)


class AuthenticationError(ThetadataError):
    """Raised when authentication fails (401 errors)."""
    def __init__(self, message: str, response: Optional[Any] = None):
        details = {"response": str(response)} if response else {}
        super().__init__(f"Authentication error: {message}", details)


class PermissionError(ThetadataError):
    """Raised when user lacks required permissions (403 errors)."""
    def __init__(self, message: str, response: Optional[Any] = None):
        details = {"response": str(response)} if response else {}
        super().__init__(f"Permission error: {message}", details)


class RateLimitError(ThetadataError):
    """Raised when API rate limit is exceeded (429 errors)."""
    def __init__(self, message: str, response: Optional[Any] = None):
        details = {"response": str(response)} if response else {}
        super().__init__(f"Rate limit error: {message}", details)


class ServiceError(ThetadataError):
    """Raised when API service encounters an error (5xx errors)."""
    def __init__(self, message: str, response: Optional[Any] = None, status_code: Optional[int] = None):
        details = {
            "response": str(response) if response else None,
            "status_code": status_code
        }
        super().__init__(f"Service error: {message}", details)


class ThetadataConnectionError(ThetadataError):
    """Raised if connection to the server fails."""
    def __init__(self, message: str, host: Optional[str] = None, port: Optional[int] = None):
        details = {"host": host, "port": port} if host and port else {}
        super().__init__(f"Connection error: {message}", details)


class ReconnectingError(ThetadataConnectionError):
    """Raised if the connection has been lost and a reconnection attempt is being made."""
