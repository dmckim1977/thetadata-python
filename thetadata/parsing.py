"""Module that parses data from the Terminal."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from io import StringIO
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import httpx
import ijson
import numpy as np
import pandas as pd
import requests

from .enums import DataType
from .exceptions import NoData, ReconnectingToServer, ResponseError, \
    ResponseParseError
from .utils import ms_to_time

HEADER_MAX_LENGTH = 300  # max length of header in characters
HEADER_FIELDS = ["id", "latency", "error_type", "error_msg", "next_page",
                 "format"]


# region CSV Responses
def process_csv_dataframe(content: bytes) -> pd.DataFrame:
    # TODO need docs

    try:
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        cols = df.columns

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

        # _check_header_errors_REST(header) # TODO handle csv response header
        return df

    except Exception as e:
        raise ResponseParseError(
            f'Failed to parse list for request. '
            f'Please send this error to support. {e}')


def get_paginated_csv_dataframe(url: str, params: dict):
    more_pages = True
    dataframes = []
    pages = 0
    # TODO add latency logging

    res = httpx.get(url, params=params, timeout=500).raise_for_status()
    logging.error(f'Requesting {res.url!r}')

    # Process csv response
    dataframes.append(process_csv_dataframe(res.content))

    while more_pages:
        pages += 1

        # Get next page, break if == 'null'
        next_page = res.headers.get('next-page')
        if next_page == 'null':
            break

        res = httpx.get(next_page, timeout=500).raise_for_status()

        # Process csv response
        dataframes.append(process_csv_dataframe(res.content))

    df = pd.concat(dataframes)
    logging.info(f'Retrieved {pages} pages.')

    return df


# endregion


def check_next_page(response):
    """
    Check if there's a next page in the API response

    Args:
        response (dict): API response containing header information

    Returns:
        str or False: Next page value if exists, False otherwise
    """
    next_page = response.get('header', {}).get('next_page')
    print(next_page)
    return next_page if next_page != 'null' else False


def get_paginated_dataframe_request(url: str, params: dict):
    next_page = True
    dataframes = []
    pages = 0

    res = httpx.get(url, params=params).raise_for_status().json()
    dataframes.append(parse_trade(res=res))

    while next_page:
        pages += 1
        next_page = check_next_page(res)

        res = httpx.get(url, params=params).raise_for_status().json()
        dataframes.append(parse_trade(res=res))

    df = pd.concat(dataframes)
    logging.info(f'Retrieved {pages} pages.')

    return df


def parse_list(res: json, name: str, dates: bool = False) -> pd.Series:
    """Parse REST response to pd.Series().
    Convert dates from int to datetime if dates == True
    Sort values when returning.

    :param res: json object
    :param name: # TODO docs
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

    :param res: json object
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
            f'Failed to parse list for request. '
            f'Please send this error to support. {e}')


# region Not Touched

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
            f"Failed to parse header for request: {url}. "
            f"Please send this error to support."
        ) from e


# def _check_body_errors(header: Header, body_data: bytes):
#     """Check for errors from the Terminal.
#
#     :raises NoData: if the server does not contain data for the request.
#     :raises ReconnectingToServer: if the connection has been lost
#     to Theta Data and a
#                                   reconnection attempt is being made/
#     :raises ResponseError: if the header indicates an error, containing a
#                            helpful error message.
#     """
#     if header.message_type == MessageType.ERROR:
#         msg = body_data.decode("utf-8")
#         if "no data" in msg.lower():
#             raise NoData(msg)
#         elif "disconnected" in msg.lower():
#             raise ReconnectingToServer(msg)
#         else:
#             raise ResponseError(msg)


def _check_header_errors_REST(header: dict):
    """Check for errors from the Terminal.

    :raises NoData: if the server does not contain data for the request.
    :raises ReconnectingToServer: if the connection has been lost to Theta Data
     and a
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


# class TickBody:
#     """Represents the body returned on Terminal
#     calls that deal with ticks."""
#
#     def __init__(self, format_tick: list[DataType], body_ticks: np.ndarray):
#         assert isinstance(format_tick, list) and isinstance(
#             body_ticks, np.ndarray
#         ), "Cannot initialize body bc ticks is not a DataFrame"
#         self.format_tick: list[DataType] = format_tick
#         self.body_ticks: np.ndarray = body_ticks
#
#     @classmethod
#     def parse(cls, request: str, header: Header, data: bytearray
#     ) -> DataFrame:
#         """Efficiently parse binary tick data.
#
#         :param request: the request that returned the body data
#         :param header: parsed header data
#         :param data: the binary response body
#         :return: a processed pandas dataframe
#         :raises ResponseParseError: if parsing failed
#         """
#         assert isinstance(
#             data, bytearray
#         ), f"Expected data to be bytearray type. Got {type(data)}"
#         _check_body_errors(header, data)
#         try:
#             tbody = cls._parse(header, data)
#             df = tbody._to_dataframe()
#             return df
#         except Exception as e:
#             raise ResponseParseError(
#                 f"Failed to parse body for request: {request}.
#                 Please send this error to support."
#             ) from e
#
#     @classmethod
#     def _parse(cls, header: Header, data: bytearray) -> TickBody:
#         assert (
#                 len(data) == header.size
#         ), f"Cannot parse body with {len(data)} bytes.
#         Expected {header.size} bytes."
#         n_cols = header.format_len
#         n_ticks = int(header.size / (header.format_len * 4))
#
#         # parse format tick
#         format: list[DataType] = []
#         for ci in range(n_cols):
#             int_ = int.from_bytes(data[ci * 4: ci * 4 + 4], "big")
#             format.append(DataType.from_code(int_))
#
#         # parse the rest of the ticks
#         # 4 byte integers w/ big endian order
#         dtype = np.dtype("int32").newbyteorder(">")
#         ticks = (
#             np.frombuffer(data, dtype=dtype, offset=(header.format_len * 4))
#             .reshape((n_ticks - 1), n_cols)
#             .byteswap()  # force native byte order
#             .newbyteorder()  # ^^
#         )
#
#         return cls(format_tick=format, body_ticks=ticks)
#
#     def _to_dataframe(self) -> DataFrame:
#         """Load this tick data into a pandas DataFrame and post process.
#
#         Post-processing modifies the columns of data w/
#         various quality-of-life
#         improvements.
#         """
#         # load ticks into DataFrame
#         df = pd.DataFrame(
#             self.body_ticks, columns=self.format_tick, copy=False
#         )
#         self._post_process(df)
#         return df
#
#     @classmethod
#     def _post_process(cls, df: DataFrame) -> None:
#         """Modify tick data w/ various quality-of-life improvements.
#
#         :param df: The DataFrame to modify in-place.
#         """
#         # remove trailing null tick if it exists
#         last_row = df.tail(1)
#         zeroes = last_row.squeeze() == 0
#         if zeroes.all():
#             # print(f"Dropping {last_row=}")
#             df.drop(last_row.index, inplace=True)
#
#         if DataType.PRICE_TYPE in df.columns:
#             # replace price type column with price multipliers
#
#             df[DataType.PRICE_TYPE] = _to_price_mul(df[DataType.PRICE_TYPE])
#
#             # multiply prices by price multipliers
#             for col in df.columns:
#                 if col.is_price():
#                     df[col] *= df[DataType.PRICE_TYPE]
#
#             # remove price type column
#             del df[DataType.PRICE_TYPE]
#
#         # convert date ints to datetime
#         if DataType.DATE in df.columns:
#             df[DataType.DATE] = pd.to_datetime(
#                 df[DataType.DATE], format="%Y%m%d"
#             )


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
            f"Failed to parse header for request: {response.url}. "
            f"Please send this error to support."
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
            f"Failed to parse header for request: {url}. "
            f"Please send this error to support."
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
            f"Failed to parse header for request: {url}. "
            f"Please send this error to support."
        ) from e

# def parse_hist_REST_stream(url, params) -> pd.DataFrame:
#     header = {}
#     row = []
#     header_format = []
#     loc = 0
#     s = requests.Session()
#     with requests.get(url, params=params, stream=True) as resp:
#         line_num = 0
#         for line in resp.iter_lines():
#             print(line)
#             line_num += 1
#             if line_num > 10: break


# class ListBody:
#     """Represents the body returned on every Terminal call that have one
#     DataType."""
#
#     def __init__(self, lst: Series):
#         assert isinstance(
#             lst, Series
#         ), "Cannot initialize body bc lst is not a Series"
#         self.lst: Series = lst
#
#     @classmethod
#     def parse(
#             cls, request: str, header: Header, data: bytes,
#             dates: bool = False
#     ) -> ListBody:
#         """Parse binary body data into an object.
#
#         :param request: the request that returned the header data
#         :param header: parsed header data
#         :param data: the binary response body
#         :param dates: whether to parse the data as date objects
#         :raises ResponseParseError: if parsing failed
#         """
#         _check_body_errors(header, data)
#         try:
#             return cls._parse(header, data, dates)
#         except Exception as e:
#             raise ResponseParseError(
#                 f"Failed to parse header for request: {request}.
#                 Please send this error to support."
#             ) from e
#
#     @classmethod
#     def _parse(
#             cls, header: Header, data: bytes, dates: bool = False
#     ) -> ListBody:
#         assert (
#                 len(data) == header.size
#         ), f"Cannot parse body with {len(data)} bytes.
#         Expected {header.size} bytes."
#
#         lst = data.decode("ascii").split(",")
#         lst = pd.Series(lst, copy=False)
#
#         if dates:
#             lst = pd.to_datetime(lst, format="%Y%m%d")
#
#         return cls(lst=lst)


# @dataclass
# class Header:
#     """Represents the header returned on every Terminal call."""
#
#     message_type: MessageType
#     id: int
#     latency: int
#     error: int
#     format_len: int
#     size: int
#
#     @classmethod
#     def parse(cls, request: str, data: bytes) -> Header:
#         """Parse binary header data into an object.
#
#         :param request: the request that returned the header data
#         :param data: raw header data, 20 bytes long
#         :raises ResponseParseError: if parsing failed
#         """
#         try:
#             return cls._parse(data)
#         except Exception as e:
#             raise ResponseParseError(
#                 f"Failed to parse header for request: {request}. "
#                 f"Please send this error to support."
#             ) from e
#
#     @classmethod
#     def _parse(cls, data: bytes) -> Header:
#         """Parse binary header data into an object.
#
#         :param data: raw header data, 20 bytes long
#         """
#         assert (
#                 len(data) == 20
#         ), f"Cannot parse header with {len(data)} bytes. Expected 20 bytes."
#         # avoid copying header data when slicing
#         data = memoryview(data)
#         """
#         Header format:
#             bytes | field
#                 2 | message type
#                 8 | id
#                 2 | latency
#                 2 | error
#                 1 | reserved / special
#                 1 | format length
#                 4 | size
#         """
#         parse_int = lambda d: int.from_bytes(d, "big")
#         # parse
#         msgtype = MessageType.from_code(parse_int(data[:2]))
#         id = parse_int(data[2:10])
#         latency = parse_int(data[10:12])
#         error = parse_int(data[12:14])
#         format_len = data[15]
#         size = parse_int(data[16:20])
#         return cls(
#             message_type=msgtype,
#             id=id,
#             latency=latency,
#             error=error,
#             format_len=format_len,
#             size=size,
#         )

# endregion
