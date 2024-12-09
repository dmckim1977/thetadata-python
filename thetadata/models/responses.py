from datetime import date as date_type
from datetime import time as time_type
from datetime import datetime as datetime_type
from io import StringIO
from typing import Any, List, Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ..utils import ms_to_time
from ..enums import EnumMapper, QuoteCondition, Exchange, TradeCondition


class ResponseHeader(BaseModel):
    """Header information from ThetaData API response."""
    format: List[str] = Field(..., description="Column format information")
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    next_page: Optional[str] = None


# region Lists, Symbols, Dates and more...
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
                date_type(year, month, day)
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

    def as_dates(self) -> List[date_type]:
        """Convert expirations to list of datetime.date objects."""
        return [
            datetime_type.strptime(str(exp), '%Y%m%d').date()
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

# endregion


# region Indices

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
                lambda row: datetime_type.combine(row['date'],
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
                lambda row: datetime_type.combine(row['date'],
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


class IndicesSnapshotsPriceRawResponse(BaseModel):
    """Internal model for raw API response."""
    header: ResponseHeader
    response: List[List[Union[int, float]]]


class IndicesSnapshotsPriceResponse(BaseModel):
    """Response model for index price snapshot data.

    Represents current price information for an index with properly formatted fields.

    :param price: The reported price of the index
    :param date: The date of the snapshot
    :param ms_of_day: Time of the snapshot
    :param datetime: Combined date and time in ET timezone
    """
    price: float = Field(..., description="The reported price of the index")
    date: date_type = Field(..., description="The date of the snapshot")
    ms_of_day: time_type = Field(..., description="Time of the snapshot")
    datetime: datetime_type = Field(...,
                                    description="Combined date and time in ET timezone")

    @classmethod
    def from_raw_response(cls,
                          raw: IndicesSnapshotsPriceRawResponse) -> 'IndicesSnapshotsPriceResponse':
        """Create formatted response from raw API response."""
        # Get column indices from header
        price_idx = raw.header.format.index('price')
        date_idx = raw.header.format.index('date')
        ms_idx = raw.header.format.index('ms_of_day')

        # Get values from first (and only) row
        row = raw.response[0]

        # Format date
        raw_date = str(row[date_idx])
        formatted_date = date_type(
            year=int(raw_date[:4]),
            month=int(raw_date[4:6]),
            day=int(raw_date[6:8])
        )

        # Format time
        formatted_time = ms_to_time(row[ms_idx])

        # Create combined datetime
        snapshot_dt = datetime_type.combine(
            formatted_date,
            formatted_time
        ).replace(tzinfo=ZoneInfo("US/Eastern"))

        return cls(
            price=row[price_idx],
            date=formatted_date,
            ms_of_day=formatted_time,
            datetime=snapshot_dt
        )

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame."""
        df = pd.DataFrame([self.model_dump()])
        df.set_index('datetime', inplace=True)
        return df


class IndicesSnapshotsOHLCRawResponse(BaseModel):
    """Internal model for raw API response."""
    header: ResponseHeader
    response: List[List[Union[int, float]]]


class IndicesSnapshotsOHLCResponse(BaseModel):
    """Response model for index OHLC snapshot data.

    Represents current OHLC information for an index with properly formatted fields.

    :param open: Opening price of the trading session
    :param high: High price of the trading session
    :param low: Low price of the trading session
    :param close: Closing price of the trading session
    :param date: The date of the snapshot
    :param ms_of_day: Time of the snapshot
    :param datetime: Combined date and time in ET timezone
    """
    open: float = Field(..., description="Opening price of the trading session")
    high: float = Field(..., description="High price of the trading session")
    low: float = Field(..., description="Low price of the trading session")
    close: float = Field(..., description="Closing price of the trading session")
    date: date_type = Field(..., description="The date of the snapshot")
    ms_of_day: time_type = Field(..., description="Time of the snapshot")
    datetime: datetime_type = Field(..., description="Combined date and time in ET timezone")

    @classmethod
    def from_raw_response(cls, raw: IndicesSnapshotsOHLCRawResponse) -> 'IndicesSnapshotsOHLCResponse':
        """Create formatted response from raw API response."""
        # Get column indices from header
        open_idx = raw.header.format.index('open')
        high_idx = raw.header.format.index('high')
        low_idx = raw.header.format.index('low')
        close_idx = raw.header.format.index('close')
        date_idx = raw.header.format.index('date')
        ms_idx = raw.header.format.index('ms_of_day')

        # Get values from first (and only) row
        row = raw.response[0]

        # Format date
        raw_date = str(row[date_idx])
        formatted_date = date_type(
            year=int(raw_date[:4]),
            month=int(raw_date[4:6]),
            day=int(raw_date[6:8])
        )

        # Format time
        formatted_time = ms_to_time(row[ms_idx])

        # Create combined datetime
        snapshot_dt = datetime_type.combine(
            formatted_date,
            formatted_time
        ).replace(tzinfo=ZoneInfo("US/Eastern"))

        return cls(
            open=row[open_idx],
            high=row[high_idx],
            low=row[low_idx],
            close=row[close_idx],
            date=formatted_date,
            ms_of_day=formatted_time,
            datetime=snapshot_dt
        )

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame."""
        df = pd.DataFrame([self.model_dump()])
        df.set_index('datetime', inplace=True)
        return df

# endregion

# region Stocks

class StockHistoricalQuoteRow(BaseModel):
    """Single row of stock historical quote data."""
    ms_of_day: int = Field(..., description="Milliseconds since midnight Eastern")
    bid_size: int = Field(..., description="The last NBBO bid size")
    bid_exchange: int = Field(..., description="The last NBBO bid exchange")
    bid: float = Field(..., description="The last NBBO bid price")
    bid_condition: int = Field(..., description="The last NBBO bid condition")
    ask_size: int = Field(..., description="The last NBBO ask size")
    ask_exchange: int = Field(..., description="The last NBBO ask exchange")
    ask: float = Field(..., description="The last NBBO ask price")
    ask_condition: int = Field(..., description="The last NBBO ask condition")
    date: int = Field(..., description="Date in YYYYMMDD format")

    @field_validator('bid_exchange', 'ask_exchange')
    @classmethod
    def validate_exchange(cls, v: int) -> Exchange:
        return Exchange(v)

    @field_validator('bid_condition', 'ask_condition')
    @classmethod
    def validate_condition(cls, v: int) -> QuoteCondition:
        return QuoteCondition(v)


class StockHistoricalQuoteResponse(BaseModel):
    """Response model for stock historical quote data."""
    data: List[StockHistoricalQuoteRow]

    @classmethod
    def from_csv(cls, csv_content: str) -> 'StockHistoricalQuoteResponse':
        """Create response model from CSV content."""
        df = pd.read_csv(StringIO(csv_content))
        rows = [StockHistoricalQuoteRow(**row) for row in
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

        # Convert enums
        enum_mapper = EnumMapper()
        df = enum_mapper.map_dataframe_enums(df, {
            'bid_exchange': 'Exchange',
            'ask_exchange': 'Exchange',
            'bid_condition': 'QuoteCondition',
            'ask_condition': 'QuoteCondition'
        })

        # Convert date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

        # Handle ms_of_day conversion
        if 'ms_of_day' in df.columns:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        # Create datetime index if we have both date and time
        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['quote_datetime'] = df.apply(
                lambda row: datetime_type.combine(row['date'],
                                                  row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('quote_datetime', inplace=True)
        # If we only have date, set it as index
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)

        return df


class StockHistoricalEODRow(BaseModel):
    """Single row of stock historical EOD data."""
    ms_of_day: int = Field(...,
                           description="Milliseconds since midnight Eastern")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    count: int = Field(..., description="Number of trades")
    date: int = Field(..., description="Date in YYYYMMDD format")


class StockHistoricalEODResponse(BaseModel):
    """Response model for stock historical EOD data."""
    data: List[StockHistoricalEODRow]

    @classmethod
    def from_csv(cls, csv_content: str) -> 'StockHistoricalEODResponse':
        """Create response model from CSV content."""
        df = pd.read_csv(StringIO(csv_content))
        rows = [StockHistoricalEODRow(**row) for row in df.to_dict('records')]
        return cls(data=rows)

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame."""
        df = pd.DataFrame([row.model_dump() for row in self.data])

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

        if 'ms_of_day' in df.columns:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['datetime'] = df.apply(
                lambda row: datetime_type.combine(row['date'],
                                                  row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('datetime', inplace=True)
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)

        return df


class StockHistoricalOHLCRow(BaseModel):
    """Single row of stock historical OHLC data."""
    ms_of_day: int = Field(...,
                           description="Milliseconds since midnight Eastern")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Number of shares traded")
    count: int = Field(..., description="Number of trades")
    date: int = Field(..., description="Date in YYYYMMDD format")


class StockHistoricalOHLCResponse(BaseModel):
    """Response model for stock historical OHLC data."""
    data: List[StockHistoricalOHLCRow]

    @classmethod
    def from_csv(cls, csv_content: str) -> 'StockHistoricalOHLCResponse':
        """Create response model from CSV content."""
        df = pd.read_csv(StringIO(csv_content))
        rows = [StockHistoricalOHLCRow(**row) for row in df.to_dict('records')]
        return cls(data=rows)

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame with proper formatting."""
        df = pd.DataFrame([row.model_dump() for row in self.data])

        # Convert date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

        # Handle ms_of_day conversion
        if 'ms_of_day' in df.columns:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        # Create datetime index if we have both date and time
        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['ohlc_datetime'] = df.apply(
                lambda row: datetime_type.combine(row['date'],
                                                  row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('ohlc_datetime', inplace=True)
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)

        return df


class StockHistoricalTradeRow(BaseModel):
    """Single row of stock historical trade data."""
    ms_of_day: int = Field(..., description="Milliseconds since midnight Eastern")
    sequence: int = Field(..., description="Exchange sequence number")
    ext_condition1: Optional[int] = Field(None, description="Additional trade condition")
    ext_condition2: Optional[int] = Field(None, description="Additional trade condition")
    ext_condition3: Optional[int] = Field(None, description="Additional trade condition")
    ext_condition4: Optional[int] = Field(None, description="Additional trade condition")
    condition: int = Field(..., description="Trade condition")
    size: int = Field(..., description="Amount of contracts traded")
    exchange: int = Field(..., description="Exchange where trade executed")
    price: float = Field(..., description="Trade price")
    condition_flags: Optional[int] = Field(None, description="Future use")
    price_flags: Optional[int] = Field(None, description="Future use")
    volume_type: Optional[int] = Field(None, description="Future use")
    records_back: int = Field(..., description="Number of trades back for cancellations/insertions")
    date: int = Field(..., description="Date in YYYYMMDD format")


class StockHistoricalTradeResponse(BaseModel):
    """Response model for stock historical trade data."""
    data: List[StockHistoricalTradeRow]

    @classmethod
    def from_csv(cls, csv_content: str) -> 'StockHistoricalTradeResponse':
        """Create response model from CSV content."""
        df = pd.read_csv(StringIO(csv_content))
        rows = [StockHistoricalTradeRow(**row) for row in df.to_dict('records')]
        return cls(data=rows)

    def to_pandas(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame with proper formatting."""
        # Create DataFrame from response data
        df = pd.DataFrame([row.model_dump() for row in self.data])

        # Convert date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

        # Handle ms_of_day conversion
        if 'ms_of_day' in df.columns:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        # Convert conditions
        df['condition'] = df['condition'].apply(
            lambda x: TradeCondition(int(x)) if not isinstance(x,
                                                               TradeCondition) else x)

        # Convert exchanges - keep as enum objects
        df['exchange'] = df['exchange'].apply(
            lambda x: Exchange(int(x)))

        # Create datetime index if we have both date and time
        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['trade_datetime'] = df.apply(
                lambda row: datetime_type.combine(row['date'],
                                                  row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('trade_datetime', inplace=True)
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)

        return df


# endregion