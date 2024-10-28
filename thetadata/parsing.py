"""Module that parses data from the Terminal."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from functools import wraps
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import httpx
import ijson
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame, Series

from .enums import DataType, MessageType
from .exceptions import NoData, ReconnectingToServer, ResponseError, \
    ResponseParseError

HEADER_MAX_LENGTH = 300  # max length of header in characters
HEADER_FIELDS = ["id", "latency", "error_type", "error_msg", "next_page",
                 "format"]


class DataHandler:
    """Handles data retrieval, pagination, and parsing for ThetaClient."""

    def __init__(self, url: str, parser: Callable):
        """
        Initialize the DataHandler.

        Args:
            base_url: Base URL for API requests
            parser: Function to parse individual response pages
        """
        self.url = url
        self.parser = parser

    def get_paginated_data(
            self,
            url: str,
            params: Dict[str, Any],
            timeout: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve and combine all pages of data from the API.

        Args:
            url: API endpoint to query
            params: Query parameters for initial request
            timeout: Request timeout in seconds

        Returns:
            Combined DataFrame of all pages
        """
        url = url
        dataframes = []
        current_params = params
        pages = 0

        while True:
            pages += 1
            logging.info(f'Handling page {pages}')
            print(f'Handling page {pages}')

            # Make request for current page
            response = httpx.get(
                url,
                params=current_params,
                timeout=timeout
            ).raise_for_status().json()

            # Parse the current page
            df = self.parser(response['header'])
            dataframes.append(df)

            # Check if there are more pages
            print(response.get('header', {}))
            next_page = response.get('header', {}).get('next_page')
            if next_page == 'null' or not next_page:
                break

            # Update URL for next page and clear params
            url = next_page
            current_params = None

        # Combine all dataframes
        if len(dataframes) == 1:
            return dataframes[0]

        return pd.concat(dataframes, axis=0, ignore_index=True)


def with_pagination(func):
    """
    Decorator to handle pagination for ThetaClient methods.

    Automatically uses DataHandler to handle pagination for any client method
    that returns paginated data.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get the endpoint and params from the decorated method
        url, params = func(self, *args, **kwargs)

        # Create DataHandler instance with appropriate parser
        handler = DataHandler(
            url=url,
            parser=parse_trade
        )

        # Get paginated data
        return handler.get_paginated_data(url, params)

    return wrapper


@dataclass
class Header:
    """Represents the header returned on every Terminal call."""

    message_type: MessageType
    id: int
    latency: int
    error: int
    format_len: int
    size: int

    @classmethod
    def parse(cls, request: str, data: bytes) -> Header:
        """Parse binary header data into an object.

        :param request: the request that returned the header data
        :param data: raw header data, 20 bytes long
        :raises ResponseParseError: if parsing failed
        """
        try:
            return cls._parse(data)
        except Exception as e:
            raise ResponseParseError(
                f"Failed to parse header for request: {request}. Please send this error to support."
            ) from e

    @classmethod
    def _parse(cls, data: bytes) -> Header:
        """Parse binary header data into an object.

        :param data: raw header data, 20 bytes long
        """
        assert (
                len(data) == 20
        ), f"Cannot parse header with {len(data)} bytes. Expected 20 bytes."
        # avoid copying header data when slicing
        data = memoryview(data)
        """
        Header format:
            bytes | field
                2 | message type
                8 | id
                2 | latency
                2 | error
                1 | reserved / special
                1 | format length
                4 | size
        """
        parse_int = lambda d: int.from_bytes(d, "big")
        # parse
        msgtype = MessageType.from_code(parse_int(data[:2]))
        id = parse_int(data[2:10])
        latency = parse_int(data[10:12])
        error = parse_int(data[12:14])
        format_len = data[15]
        size = parse_int(data[16:20])
        return cls(
            message_type=msgtype,
            id=id,
            latency=latency,
            error=error,
            format_len=format_len,
            size=size,
        )


def parse_list(res: json, name: str, dates: bool = False) -> pd.Series:
    """Parse REST response to pd.Series().
    Convert dates from int to datetime if dates == True
    Sort values when returning.

    :param response: json object
    :param dates: whether to parse the data as date objects. Format YYYYMMDD
    :raises ResponseParseError: if parsing failed
    """

    header = res['header']
    try:
        lst = pd.Series(res['response'], name=name)
        _check_header_errors_REST(header)

        if dates:
            try:
                return pd.to_datetime(lst, format="%Y%m%d").sort_values()
            except Exception as e:
                logging.error(f"Failed to parse date for {name}. {e}")
        else:
            return lst.sort_values()
    except Exception as e:
        raise ResponseParseError(
            f'Failed to parse list for request: {name}. '
            f'Please send this error to support. {e}')


def parse_trade(res: json) -> pd.DataFrame:
    """Parse REST response to pd.Series().
    Convert dates from int to datetime if dates == True
    Sort values when returning.

    :param response: json object
    :param dates: whether to parse the data as date objects. Format YYYYMMDD
    :raises ResponseParseError: if parsing failed
    """
    # TODO convert ms_of_day
    header = res['header']
    cols = header['format']
    try:
        df = pd.DataFrame(res['response'], columns=cols)
        if 'strike' in cols:
            df['strike'] = df['strike'] / 1000

        if 'expiration' in cols:
            df['expiration'] = pd.to_datetime(df['expiration'],
                                              format="%Y%m%d")
            df.set_index('expiration', inplace=True)

        if {'date', 'ms_of_day'}.issubset(df.columns):
            df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)
            df['trade_datetime'] = df.apply(
                lambda row: datetime.combine(row['date'],
                                             row['ms_of_day']).replace(
                    tzinfo=ZoneInfo("US/Eastern")
                ),
                axis=1
            )
            df.set_index('trade_datetime', inplace=True)

        elif 'date' in cols:
            df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
            df.set_index('date', inplace=True)

        elif 'ms_of_day' in cols:
            df['ms_of_day'] = df['ms_of_day'].apply(ms_to_time)

        _check_header_errors_REST(header)
        return df

    except Exception as e:
        raise ResponseParseError(
            f'Failed to parse list for request. Please send this error to support. {e}')


def ms_to_time(ms: int) -> time:
    """Convert milliseconds since midnight to time object."""
    try:
        seconds = ms // 1000
        microseconds = (ms % 1000) * 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return dt_time(hour=hours, minute=minutes, second=seconds,
                       microsecond=microseconds)
    except Exception as e:
        return dt_time(hour=0, minute=0, second=0, microsecond=0)
        logging.error(f"Failed to convert milliseconds: {ms} to time. {e}")


def time_to_ms(time_str):
    # Split the time string by colons
    parts = time_str.split(':')

    if len(parts) == 3:  # H:M:S format
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:  # H:M format
        hours, minutes = map(int, parts)
        seconds = 0
    else:
        raise ValueError("Invalid time format. Use H:M:S or H:M")

    # Convert to milliseconds
    total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
    logging.info(total_ms)
    return total_ms


########################## Not Touched ################################

def parse_header_REST(response: requests.Response, header_string: str) -> dict:
    """Parse JSON header data into an object.

    :param response: the full requests.Response object
    :param header_string: header data in a string of JSON format
    :raises ResponseParseError: if parsing failed
    """
    # make sure to show the first URL in case of redirects
    url = response.history[0].url if response.history else response.url
    try:
        return json.loads(header_string)
    except Exception as e:
        raise ResponseParseError(
            f"Failed to parse header for request: {url}. Please send this error to support."
        ) from e


def _check_body_errors(header: Header, body_data: bytes):
    """Check for errors from the Terminal.

    :raises NoData: if the server does not contain data for the request.
    :raises ReconnectingToServer: if the connection has been lost to Theta Data and a
                                  reconnection attempt is being made/
    :raises ResponseError: if the header indicates an error, containing a
                           helpful error message.
    """
    if header.message_type == MessageType.ERROR:
        msg = body_data.decode("utf-8")
        if "no data" in msg.lower():
            raise NoData(msg)
        elif "disconnected" in msg.lower():
            raise ReconnectingToServer(msg)
        else:
            raise ResponseError(msg)


def _check_header_errors_REST(header: dict):
    """Check for errors from the Terminal.

    :raises NoData: if the server does not contain data for the request.
    :raises ReconnectingToServer: if the connection has been lost to Theta Data and a
                                  reconnection attempt is being made/
    :raises ResponseError: if the header indicates an error, containing a
                           helpful error message.
    """
    error = header.get("error_type", None)
    if error is not None:
        if header.get("error_type").lower() != "null":
            msg = header["error_msg"]
            if "no data" in msg.lower():
                raise NoData(msg)
            elif "disconnected" in msg.lower():
                raise ReconnectingToServer(msg)
            else:
                raise ResponseError(msg)


# map price types to price multipliers
_pt_to_price_mul = [
    0,
    0.000000001,
    0.00000001,
    0.0000001,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    1,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
]

# Vectorized function that maps price types to price multipliers
_to_price_mul = np.vectorize(lambda pt: _pt_to_price_mul[pt], otypes=[float])


class TickBody:
    """Represents the body returned on Terminal calls that deal with ticks."""

    def __init__(self, format_tick: list[DataType], body_ticks: np.ndarray):
        assert isinstance(format_tick, list) and isinstance(
            body_ticks, np.ndarray
        ), "Cannot initialize body bc ticks is not a DataFrame"
        self.format_tick: list[DataType] = format_tick
        self.body_ticks: np.ndarray = body_ticks

    @classmethod
    def parse(cls, request: str, header: Header, data: bytearray) -> DataFrame:
        """Efficiently parse binary tick data.

        :param request: the request that returned the body data
        :param header: parsed header data
        :param data: the binary response body
        :return: a processed pandas dataframe
        :raises ResponseParseError: if parsing failed
        """
        assert isinstance(
            data, bytearray
        ), f"Expected data to be bytearray type. Got {type(data)}"
        _check_body_errors(header, data)
        try:
            tbody = cls._parse(header, data)
            df = tbody._to_dataframe()
            return df
        except Exception as e:
            raise ResponseParseError(
                f"Failed to parse body for request: {request}. Please send this error to support."
            ) from e

    @classmethod
    def _parse(cls, header: Header, data: bytearray) -> TickBody:
        assert (
                len(data) == header.size
        ), f"Cannot parse body with {len(data)} bytes. Expected {header.size} bytes."
        n_cols = header.format_len
        n_ticks = int(header.size / (header.format_len * 4))

        # parse format tick
        format: list[DataType] = []
        for ci in range(n_cols):
            int_ = int.from_bytes(data[ci * 4: ci * 4 + 4], "big")
            format.append(DataType.from_code(int_))

        # parse the rest of the ticks
        # 4 byte integers w/ big endian order
        dtype = np.dtype("int32").newbyteorder(">")
        ticks = (
            np.frombuffer(data, dtype=dtype, offset=(header.format_len * 4))
            .reshape((n_ticks - 1), n_cols)
            .byteswap()  # force native byte order
            .newbyteorder()  # ^^
        )

        return cls(format_tick=format, body_ticks=ticks)

    def _to_dataframe(self) -> DataFrame:
        """Load this tick data into a pandas DataFrame and post process.

        Post-processing modifies the columns of data w/ various quality-of-life
        improvements.
        """
        # load ticks into DataFrame
        df = pd.DataFrame(
            self.body_ticks, columns=self.format_tick, copy=False
        )
        self._post_process(df)
        return df

    @classmethod
    def _post_process(cls, df: DataFrame) -> None:
        """Modify tick data w/ various quality-of-life improvements.

        :param df: The DataFrame to modify in-place.
        """
        # remove trailing null tick if it exists
        last_row = df.tail(1)
        zeroes = last_row.squeeze() == 0
        if zeroes.all():
            # print(f"Dropping {last_row=}")
            df.drop(last_row.index, inplace=True)

        if DataType.PRICE_TYPE in df.columns:
            # replace price type column with price multipliers

            df[DataType.PRICE_TYPE] = _to_price_mul(df[DataType.PRICE_TYPE])

            # multiply prices by price multipliers
            for col in df.columns:
                if col.is_price():
                    df[col] *= df[DataType.PRICE_TYPE]

            # remove price type column
            del df[DataType.PRICE_TYPE]

        # convert date ints to datetime
        if DataType.DATE in df.columns:
            df[DataType.DATE] = pd.to_datetime(
                df[DataType.DATE], format="%Y%m%d"
            )


def parse_flexible_REST(response: requests.Response) -> pd.DataFrame:
    """
    Flexible parsing function that uses a python dictionary as an intermediary
    between json string and pandas dataframe.
    """
    response_dict = response.json()
    _check_header_errors_REST(response["header"])
    cols = [DataType.from_string(name=col) for col in
            response_dict['header']['format']]
    rows = response_dict['response']
    df = pd.DataFrame(rows, columns=cols)
    print(df)
    if DataType.DATE in df.columns:
        df[DataType.DATE] = pd.to_datetime(
            df[DataType.DATE], format="%Y%m%d"
        )
    try:
        return df
    except Exception as e:
        raise ResponseParseError(
            f"Failed to parse header for request: {response.url}. Please send this error to support."
        ) from e


def parse_hist_REST(response: requests.Response) -> pd.DataFrame:
    resp_split = response.text.split('"response": ')
    to_lstrip = '"header": \t\n'
    to_rstrip = ", \t\n"
    header_str = resp_split[0][1:].lstrip(to_lstrip).rstrip(to_rstrip)
    header = json.loads(header_str)
    _check_header_errors_REST(header)
    cols = [DataType.from_string(name=col) for col in header['format']]
    rows = pd.read_json(resp_split[1][:-1], orient="table")
    df = pd.DataFrame(rows, columns=cols)
    if DataType.DATE in df.columns:
        df[DataType.DATE] = pd.to_datetime(
            df[DataType.DATE], format="%Y%m%d"
        )
    url = response.history[0].url if response.history else response.url
    try:
        return df
    except Exception as e:
        raise ResponseParseError(
            f"Failed to parse header for request: {url}. Please send this error to support."
        ) from e


def parse_hist_REST_stream_ijson(url, params) -> pd.DataFrame:
    url = url + '?' + urlencode(params)
    f = urlopen(url)
    header = {}
    row = []
    header_format = []
    loc = 0
    for prefix, event, value in ijson.parse(f, use_float=True):
        if prefix == "response.item.item":
            row.append(value)

        elif prefix == "response.item" and event == "end_array":
            df.loc[loc] = row
            loc += 1
            row = []

        elif prefix == "header.format.item":
            header_format.append(value)

        elif prefix[:6] == "header" and len(prefix) > 6:
            header[prefix[7:]] = value

        elif event == "map_key" and value == "response":
            header["format"] = header_format
            _check_header_errors_REST(header)
            cols = [DataType.from_string(name=col) for col in header['format']]
            df = pd.DataFrame(columns=cols)

    if DataType.DATE in df.columns:
        df[DataType.DATE] = pd.to_datetime(
            df[DataType.DATE], format="%Y%m%d"
        )
    try:
        return df
    except Exception as e:
        raise ResponseParseError(
            f"Failed to parse header for request: {url}. Please send this error to support."
        ) from e


def parse_hist_REST_stream(url, params) -> pd.DataFrame:
    header = {}
    row = []
    header_format = []
    loc = 0
    s = requests.Session()
    with requests.get(url, params=params, stream=True) as resp:
        line_num = 0
        for line in resp.iter_lines():
            print(line)
            line_num += 1
            if line_num > 10: break


class ListBody:
    """Represents the body returned on every Terminal call that have one DataType."""

    def __init__(self, lst: Series):
        assert isinstance(
            lst, Series
        ), "Cannot initialize body bc lst is not a Series"
        self.lst: Series = lst

    @classmethod
    def parse(
            cls, request: str, header: Header, data: bytes, dates: bool = False
    ) -> ListBody:
        """Parse binary body data into an object.

        :param request: the request that returned the header data
        :param header: parsed header data
        :param data: the binary response body
        :param dates: whether to parse the data as date objects
        :raises ResponseParseError: if parsing failed
        """
        _check_body_errors(header, data)
        try:
            return cls._parse(header, data, dates)
        except Exception as e:
            raise ResponseParseError(
                f"Failed to parse header for request: {request}. Please send this error to support."
            ) from e

    @classmethod
    def _parse(
            cls, header: Header, data: bytes, dates: bool = False
    ) -> ListBody:
        assert (
                len(data) == header.size
        ), f"Cannot parse body with {len(data)} bytes. Expected {header.size} bytes."

        lst = data.decode("ascii").split(",")
        lst = pd.Series(lst, copy=False)

        if dates:
            lst = pd.to_datetime(lst, format="%Y%m%d")

        return cls(lst=lst)
