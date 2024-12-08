from datetime import date, datetime
from io import StringIO
from typing import Any, List, Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ..utils import ms_to_time


class ResponseHeader(BaseModel):
    """Header information from ThetaData API response."""
    format: List[str] = Field(..., description="Column format information")
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    next_page: Optional[str] = None


class StockHistoricalEODResponse(BaseModel):
    """Response model for stock historical EOD data.

    The response data is a list of lists where each inner list represents
    a row of data with values corresponding to the columns defined in header.format
    """
    header: ResponseHeader
    response: List[List[Union[int, float, str]]]

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame with proper formatting."""
        # Create DataFrame using header format as columns
        df = pd.DataFrame(self.response, columns=self.header.format)

        # Convert date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        # Handle ms_of_day if present
        if 'ms_of_day' in df.columns:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        # Handle ms_of_day2 if present
        if 'ms_of_day2' in df.columns:
            df['ms_of_day2'] = df['ms_of_day2'].apply(ms_to_time)

        # Create datetime columns if we have both date and time
        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['eod_datetime'] = df.apply(
                lambda row: datetime.combine(row['date'],
                                             row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('eod_datetime', inplace=True)

        # If we only have date, set it as index
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)

        return df

    @field_validator('response')
    @classmethod
    def validate_response_format(cls, v: List[List[Any]],
                                 info: ValidationInfo) -> List[List[Any]]:
        """Validate response data matches header format."""
        if not hasattr(info.data, 'header'):
            return v

        header_format = info.data.header.format
        expected_length = len(header_format)

        # Check each row has correct number of columns
        for i, row in enumerate(v):
            if len(row) != expected_length:
                raise ValueError(
                    f"Row {i} has {len(row)} values but header specifies {expected_length} columns"
                )

        return v


class ExpirationsResponse(BaseModel):
    """Response model for option expirations data.

    Acts like a list of expiration dates while providing helper methods
    for different formats.
    """
    header: ResponseHeader
    response: List[int] = Field(...,
                                description="List of expiration dates in YYYYMMDD format")

    @field_validator('response')
    @classmethod
    def validate_expirations(cls, v: List[int]) -> List[int]:
        """Validate each expiration date."""
        for exp in v:
            str_date = str(exp)
            if len(str_date) != 8:
                raise ValueError(f"Expiration {exp} not in YYYYMMDD format")
            try:
                year = int(str_date[:4])
                month = int(str_date[4:6])
                day = int(str_date[6:8])
                date(year, month, day)
            except ValueError:
                raise ValueError(f"Invalid expiration date: {exp}")
        return sorted(v)  # Return sorted list for consistency

    def as_pandas(self) -> pd.Series:
        """Convert expirations to pandas Series with datetime index."""
        return pd.Series(
            index=pd.to_datetime(self.response, format='%Y%m%d'),
            data=self.response,
            name="expirations"
        )

    def as_dates(self) -> List[datetime.date]:
        """Convert expirations to list of datetime.date objects."""
        return [
            datetime.strptime(str(exp), '%Y%m%d').date()
            for exp in self.response
        ]

    def __iter__(self):
        """Allow direct iteration over expiration dates."""
        return iter(self.response)

    def __len__(self):
        """Return number of expirations."""
        return len(self.response)

    def __getitem__(self, idx):
        """Allow indexing like a list."""
        return self.response[idx]

    def __contains__(self, item):
        """Allow 'in' operator."""
        return item in self.response

    def __repr__(self):
        """Show as a list of dates."""
        return repr(self.response)


class IndicesHistoricalEODRow(BaseModel):
    """Single row of indices historical EOD data."""
    ms_of_day: int = Field(...,
                           description="Milliseconds since midnight Eastern")
    ms_of_day2: int = Field(...,
                            description="Milliseconds since midnight Eastern")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    date: int = Field(..., description="Date in YYYY-MM-DD format")


class IndicesHistoricalEODResponse(BaseModel):
    """Response model for indices historical EOD data."""
    data: List[IndicesHistoricalEODRow]

    @classmethod
    def from_csv(cls, csv_content: str) -> 'IndicesHistoricalEODResponse':
        """Create response model from CSV content."""
        df = pd.read_csv(StringIO(csv_content))
        rows = [IndicesHistoricalEODRow(**row) for row in
                df.to_dict('records')]
        return cls(data=rows)

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame with proper formatting."""
        # Create DataFrame from response data
        df = pd.DataFrame([row.model_dump() for row in self.data])

        # Convert date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

        # Handle ms_of_day conversions
        if 'ms_of_day' in df.columns:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        if 'ms_of_day2' in df.columns:
            df['ms_of_day2'] = df['ms_of_day2'].apply(ms_to_time)

        # Create datetime columns if we have both date and time
        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['eod_datetime'] = df.apply(
                lambda row: datetime.combine(row['date'],
                                             row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('eod_datetime', inplace=True)
        # If we only have date, set it as index
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)

        return df


class IndicesHistoricalPriceRow(BaseModel):
    """Single row of indices historical price data."""
    ms_of_day: int = Field(
        ...,
        description="Milliseconds since midnight Eastern"
    )
    price: float = Field(
        ...,
        description="The reported price of the index"
    )
    date: int = Field(
        ...,
        description="Date in YYYYMMDD format"
    )


class IndicesHistoricalPriceResponse(BaseModel):
    """Response model for indices historical price data."""
    data: List[IndicesHistoricalPriceRow]

    @classmethod
    def from_csv(cls, csv_content: str) -> 'IndicesHistoricalPriceResponse':
        """Create response model from CSV content."""
        df = pd.read_csv(StringIO(csv_content))
        rows = [IndicesHistoricalPriceRow(**row) for row in
                df.to_dict('records')]
        return cls(data=rows)

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame with proper formatting.

        Returns:
            DataFrame with properly formatted columns:
            - date converted to datetime.date
            - ms_of_day converted to time object
            - Creates a datetime index in ET timezone if both date and ms_of_day present
        """
        # Create DataFrame from response data
        df = pd.DataFrame([row.model_dump() for row in self.data])

        # Convert date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

        # Handle ms_of_day conversion
        if 'ms_of_day' in df.columns:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        # Create datetime index if we have both date and time
        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['price_datetime'] = df.apply(
                lambda row: datetime.combine(row['date'],
                                             row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('price_datetime', inplace=True)
        # If we only have date, set it as index
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)

        return df


class IndicesSnapshotsPriceResponse(BaseModel):
    """Response model for index price snapshot data.

    Represents current price information for an index.

    :param ms_of_day: Milliseconds since midnight Eastern
    :param price: The reported price of the index
    :param date: Date in YYYYMMDD format
    """
    ms_of_day: int = Field(
        ...,
        description="Milliseconds since midnight Eastern"
    )
    price: float = Field(
        ...,
        description="The reported price of the index"
    )
    date: int = Field(
        ...,
        description="Date in YYYYMMDD format"
    )

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame with proper formatting."""
        # Create single row DataFrame
        df = pd.DataFrame([self.model_dump()])

        # Convert date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

        # Handle ms_of_day conversion
        if 'ms_of_day' in df.columns:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        # Create datetime index
        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['snapshot_datetime'] = df.apply(
                lambda row: datetime.combine(row['date'],
                                             row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('snapshot_datetime', inplace=True)
        # If we only have date, set it as index
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)

        return df
