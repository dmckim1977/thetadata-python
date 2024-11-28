from pydantic import BaseModel, Field, field_validator
from datetime import date
import pandas as pd
import httpx
from typing import List


class StockHistoricalEODReportParams(BaseModel):
    start_date: int = Field(..., description="Start date in YYYYMMDD format")
    end_date: int = Field(..., description="End date in YYYYMMDD format")
    root: str = Field(..., min_length=1, description="Stock root symbol")

    @field_validator('start_date', 'end_date')
    def validate_date_format(cls, value: int) -> int:
        str_date = str(value)
        if len(str_date) != 8:
            raise ValueError("Date must be in YYYYMMDD format")
        try:
            year = int(str_date[:4])
            month = int(str_date[4:6])
            day = int(str_date[6:])
            date(year, month, day)
        except ValueError:
            raise ValueError("Invalid date")
        return value

    @field_validator('end_date')
    def validate_end_date_after_start(cls, value: int, values: dict) -> int:
        if 'start_date' in values and value < values['start_date']:
            raise ValueError("end_date must be after start_date")
        return value