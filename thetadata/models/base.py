from datetime import date
from typing import Any
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class DateRange(BaseModel):
    """Base model for date range parameters.

    :param start_date: Start date in YYYYMMDD format
    :param end_date: End date in YYYYMMDD format
    """
    start_date: int = Field(..., description="Start date in YYYYMMDD format")
    end_date: int = Field(..., description="End date in YYYYMMDD format")

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: int, info: ValidationInfo) -> int:
        """Validate date is in YYYYMMDD format."""
        str_date = str(v)
        if len(str_date) != 8:
            raise ValueError(f"{info.field_name} must be in YYYYMMDD format")
        try:
            year = int(str_date[:4])
            month = int(str_date[4:6])
            day = int(str_date[6:8])
            date(year, month, day)  # Validate date is real
            return v
        except ValueError:
            raise ValueError(f"Invalid date in {info.field_name}: {v}")

    @field_validator('end_date')
    @classmethod
    def validate_end_date_after_start(cls, v: int,
                                      info: ValidationInfo) -> int:
        """Validate end_date is after start_date."""
        start_date = info.data.get('start_date')
        if start_date is not None and v < start_date:
            raise ValueError("end_date must be after start_date")
        return v


class RootMixin(BaseModel):
    """Base model for endpoints that require a root/ticker symbol.

    :param root: The root/ticker symbol (e.g., 'AAPL', 'SPY')
    """
    root: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock/Option root symbol (ticker)"
    )

    @field_validator('root')
    @classmethod
    def validate_root(cls, v: str) -> str:
        """Validate and transform root to uppercase."""
        if not v.strip():
            raise ValueError("Root cannot be empty or just whitespace")
        return v.strip().upper()


class ExpirationMixin(BaseModel):
    """Base model for endpoints that require an expiration date.

    :param exp: Expiration date in YYYYMMDD format
    """
    exp: int = Field(
        ...,
        description="Expiration date in YYYYMMDD format",
        ge=19000101,  # Basic sanity check for dates
        le=29991231
    )

    @field_validator('exp')
    @classmethod
    def validate_expiration(cls, v: int) -> int:
        """Validate expiration date format."""
        str_date = str(v)
        if len(str_date) != 8:
            raise ValueError("Expiration must be in YYYYMMDD format")
        try:
            year = int(str_date[:4])
            month = int(str_date[4:6])
            day = int(str_date[6:8])
            date(year, month, day)  # Validate date is real
            return v
        except ValueError:
            raise ValueError(f"Invalid expiration date: {v}")