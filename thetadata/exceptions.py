"""Module containing custom exceptions for Thetadata API."""
from typing import Any, Optional, Dict

class ThetadataError(Exception):
    """Base exception class for all Thetadata exceptions.

    :param message: Human readable error message
    :param details: Optional dictionary containing additional error details
    :param error_code: Optional error code from Thetadata API
    """
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[int] = None
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self) -> str:
        base_message = self.message
        if self.error_code:
            base_message = f"[Error {self.error_code}] {base_message}"
        if self.details:
            return f"{base_message}\nDetails: {self.details}"
        return base_message


class NoImplementationError(ThetadataError):
    """Raised when there is no implementation for the requested endpoint (404).

    This error occurs when either:
    - The request being made is invalid
    - Using an outdated Theta Terminal version
    """
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "No implementation found for this request. Check if the request is valid or update Theta Terminal."
        super().__init__(message=message, details=details, error_code=404)


class OSLimitError(ThetadataError):
    """Raised when the operating system is throttling requests (429).

    This occurs when making a large amount of small low latency requests.
    The request should be retried until this error no longer occurs.
    """
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Operating system is throttling requests. Retry the request."
        super().__init__(message=message, details=details, error_code=429)


class GeneralError(ThetadataError):
    """Raised for general/unspecified errors from the API (470)."""
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "A general error occurred"
        super().__init__(message=message, details=details, error_code=470)


class PermissionError(ThetadataError):
    """Raised when account lacks required permissions (471)."""
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Your account does not have the required permissions"
        super().__init__(message=message, details=details, error_code=471)


class NoDataError(ThetadataError):
    """Raised when no data is found for the specified request (472)."""
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "No data found for the specified request"
        super().__init__(message=message, details=details, error_code=472)


class InvalidParamsError(ThetadataError):
    """Raised when request parameters/syntax are invalid (473).

    This may be resolved by updating to the latest version of Theta Terminal.
    """
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Invalid parameters or syntax in request"
        super().__init__(message=message, details=details, error_code=473)


class DisconnectedError(ThetadataError):
    """Raised when connection to Theta Data MDDS is lost (474)."""
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Connection lost to Theta Data MDDS"
        super().__init__(message=message, details=details, error_code=474)


class TerminalParseError(ThetadataError):
    """Raised when there's an issue parsing the request after receipt (475)."""
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Terminal failed to parse the received request"
        super().__init__(message=message, details=details, error_code=475)


class WrongIPError(ThetadataError):
    """Raised when IP address doesn't match initial request IP (476).

    Requests must use the same IP (cannot switch between 127.0.0.1 and localhost).
    """
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "IP address mismatch. Use consistent IP for requests"
        super().__init__(message=message, details=details, error_code=476)


class NoPageError(ThetadataError):
    """Raised when requested page doesn't exist or has expired (477)."""
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Page does not exist or has expired"
        super().__init__(message=message, details=details, error_code=477)


class LargeRequestError(ThetadataError):
    """Raised when request asks for too much data (570)."""
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Request exceeds maximum allowed data size"
        super().__init__(message=message, details=details, error_code=570)


class ServerStartingError(ThetadataError):
    """Raised when server is intentionally restarting (571)."""
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Server is currently restarting"
        super().__init__(message=message, details=details, error_code=571)


class UncaughtError(ThetadataError):
    """Raised for uncaught server errors (572).

    When encountering this error, contact support with the exact request details.
    """
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = message or "Uncaught server error occurred. Contact support with request details"
        super().__init__(message=message, details=details, error_code=572)


class ServiceError(ThetadataError):
    """Raised when the API service encounters an error (5xx errors).

    This is a general server-side error for HTTP 5xx status codes that don't
    map to specific Thetadata error codes.
    """
    def __init__(
        self,
        message: str = "Service error occurred",
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None
    ):
        if status_code:
            details = details or {}
            details["status_code"] = status_code
        super().__init__(message=message, details=details)


# HTTP Status Code to Exception mapping
ERROR_CODE_MAP = {
    404: NoImplementationError,
    429: OSLimitError,
    470: GeneralError,
    471: PermissionError,
    472: NoDataError,
    473: InvalidParamsError,
    474: DisconnectedError,
    475: TerminalParseError,
    476: WrongIPError,
    477: NoPageError,
    570: LargeRequestError,
    571: ServerStartingError,
    572: UncaughtError
}