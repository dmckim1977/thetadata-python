"""Tests for Thetadata exception handling."""
import pytest
import httpx
from typing import Dict, Type

from thetadata.exceptions import (
    ThetadataError,
    NoImplementationError,
    OSLimitError,
    GeneralError,
    PermissionError,
    NoDataError,
    InvalidParamsError,
    DisconnectedError,
    TerminalParseError,
    WrongIPError,
    NoPageError,
    LargeRequestError,
    ServerStartingError,
    UncaughtError,
    ERROR_CODE_MAP,
    ServiceError,
)

# Test data for all error types
ERROR_TEST_CASES = [
    (404, NoImplementationError, "No implementation found"),
    (429, OSLimitError, "Operating system is throttling requests"),
    (470, GeneralError, "A general error occurred"),
    (471, PermissionError,
     "Your account does not have the required permissions"),
    (472, NoDataError, "No data found for the specified request"),
    (473, InvalidParamsError, "Invalid parameters or syntax in request"),
    (474, DisconnectedError, "Connection lost to Theta Data MDDS"),
    (475, TerminalParseError, "Terminal failed to parse the received request"),
    (476, WrongIPError, "IP address mismatch"),
    (477, NoPageError, "Page does not exist or has expired"),
    (570, LargeRequestError, "Request exceeds maximum allowed data size"),
    (571, ServerStartingError, "Server is currently restarting"),
    (572, UncaughtError, "Uncaught server error occurred")
]


@pytest.mark.parametrize("error_code,exception_class,expected_message",
                         ERROR_TEST_CASES)
def test_exception_basics(error_code: int,
                          exception_class: Type[ThetadataError],
                          expected_message: str):
    """Test basic exception properties and behavior."""
    # Test with default message
    exc = exception_class()
    assert exc.error_code == error_code
    assert expected_message in str(exc)

    # Test with custom message
    custom_msg = "Custom error message"
    exc = exception_class(message=custom_msg)
    assert exc.error_code == error_code
    assert custom_msg in str(exc)

    # Test with details
    details = {"param": "value"}
    exc = exception_class(details=details)
    assert exc.details == details
    assert "Details" in str(exc)

    assert "param" in str(exc)


def test_error_code_map_completeness():
    """Verify ERROR_CODE_MAP contains all expected error codes."""
    expected_codes = {code for code, _, _ in ERROR_TEST_CASES}
    actual_codes = set(ERROR_CODE_MAP.keys())
    assert expected_codes == actual_codes, "Missing or extra error codes in ERROR_CODE_MAP"


class MockResponse:
    """Mock httpx.Response for testing."""

    def __init__(self, status_code: int, json_data: Dict = None,
                 text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._text = text

    def json(self):
        return self._json_data

    @property
    def text(self):
        return self._text


class MockHTTPError(httpx.HTTPError):
    """Mock httpx.HTTPError for testing."""

    def __init__(self, response: MockResponse):
        self.response = response


@pytest.mark.parametrize("error_code,exception_class,_", ERROR_TEST_CASES)
def test_handle_http_error_mapping(theta_client, error_code: int,
                                   exception_class: Type[ThetadataError], _):
    """Test that _handle_http_error correctly maps status codes to exceptions."""
    response = MockResponse(
        status_code=error_code,
        json_data={"error": "test error"},
        text="Error occurred"
    )
    error = MockHTTPError(response)

    with pytest.raises(exception_class) as exc_info:
        theta_client._handle_http_error(error, params={"test": "param"})

    assert exc_info.value.error_code == error_code
    assert "test" in exc_info.value.details["params"]


def test_handle_http_error_with_json_response(theta_client):
    """Test error handling with JSON response body."""
    error_data = {"error": "test error", "details": "more info"}
    response = MockResponse(
        status_code=472,
        json_data=error_data
    )
    error = MockHTTPError(response)

    with pytest.raises(NoDataError) as exc_info:
        theta_client._handle_http_error(error)

    assert error_data == exc_info.value.details["response"]


def test_handle_http_error_with_text_response(theta_client):
    """Test error handling with text response body."""
    error_text = "Error message"
    response = MockResponse(
        status_code=473,
        json_data=None,  # Ensure json() will fail
        text=error_text
    )
    error = MockHTTPError(response)

    with pytest.raises(InvalidParamsError) as exc_info:
        theta_client._handle_http_error(error)

    # Check that the error text is in the response details
    assert error_text == exc_info.value.details["response"]


def test_handle_http_error_with_fallback(theta_client):
    """Test error handling when both JSON and text extraction fail."""

    class BrokenResponse(MockResponse):
        def json(self):
            raise ValueError("JSON decode error")

        @property
        def text(self):
            raise ValueError("Text decode error")

    response = BrokenResponse(status_code=474)
    error = MockHTTPError(response)

    with pytest.raises(DisconnectedError) as exc_info:
        theta_client._handle_http_error(error)

    assert isinstance(exc_info.value.details["response"], str)


def test_unknown_error_code(theta_client):
    """Test handling of unknown error codes."""
    response = MockResponse(
        status_code=999,
        text="Unknown error"
    )
    error = MockHTTPError(response)

    with pytest.raises(
            ServiceError if error.response.status_code >= 500 else ThetadataError) as exc_info:
        theta_client._handle_http_error(error)

    assert exc_info.value.details["status_code"] == 999


def test_error_with_empty_details():
    """Test that exceptions handle empty/None details gracefully."""
    exc = NoDataError(details=None)
    assert exc.details == {}
    assert str(exc) == f"[Error 472] No data found for the specified request"


def test_error_inheritance():
    """Verify all exceptions inherit from ThetadataError."""
    for _, exception_class, _ in ERROR_TEST_CASES:
        assert issubclass(exception_class, ThetadataError)