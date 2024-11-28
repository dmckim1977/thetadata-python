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