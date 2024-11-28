"""Module that contains custom exceptions."""
import enum
from typing import Any, Optional


class ResponseParseError(Exception):
    """Raised if the API failed to parse a Terminal response."""


class _EnumParseError(Exception):
    """Raised when a value cannot be parsed into an associated enum member."""

    def __init__(self, value: Any, enm: Any):
        """Create a new `EnumParseError`

        :param value: The value that cannot be parsed.
        :param enm: The enum.
        """
        assert issubclass(
            enm, enum.Enum
        ), "Cannot create an EnumParseError with a non-enum."
        msg = f"Value {value} cannot be parsed into a {enm.__name__}!"
        super().__init__(msg)


class ResponseError(Exception):
    """Raised if there is an error in the body of a response."""


class NoData(Exception):
    """Raised if no data is available for this request."""


class ReconnectingToServer(Exception):
    """Raised if the connection has been lost to Theta Data and a reconnection attempt is being made."""


class ThetadataValidationError(Exception):
    """Exception raised when request parameters fail validation."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message

    @classmethod
    def from_pydantic(cls, pydantic_error) -> 'ValidationError':
        """Convert a Pydantic ValidationError to our custom ValidationError.

        :param pydantic_error: The original Pydantic ValidationError
        :return: Our custom ValidationError with formatted details
        """
        details = {}
        for error in pydantic_error.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            details[field] = error["msg"]
        return cls("Validation failed", details)

    @classmethod
    def for_date_format(cls, field: str, value: Any) -> 'ValidationError':
        """Create a ValidationError for date format issues.

        :param field: Name of the field with the invalid date
        :param value: The invalid value that was provided
        :return: Formatted ValidationError
        """
        return cls(
            "Invalid date format",
            {field: f"Expected YYYYMMDD format, got {value}"}
        )

    @classmethod
    def for_invalid_range(
            cls,
            field: str,
            value: Any,
            min_value: Optional[Any] = None,
            max_value: Optional[Any] = None
    ) -> 'ValidationError':
        """Create a ValidationError for range validation issues.

        :param field: Name of the field with the invalid value
        :param value: The invalid value that was provided
        :param min_value: Optional minimum allowed value
        :param max_value: Optional maximum allowed value
        :return: Formatted ValidationError
        """
        range_str = ""
        if min_value is not None and max_value is not None:
            range_str = f"between {min_value} and {max_value}"
        elif min_value is not None:
            range_str = f"greater than or equal to {min_value}"
        elif max_value is not None:
            range_str = f"less than or equal to {max_value}"

        return cls(
            "Value out of allowed range",
            {field: f"Expected value {range_str}, got {value}"}
        )
