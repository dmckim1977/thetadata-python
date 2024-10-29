"""Module that contains Theta Client class."""
import logging
import os
import shutil
import socket
import subprocess
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Optional, get_args

import httpx
import pandas as pd
from tqdm import tqdm

from . import terminal
from .enums import *
from .literals import Rate, SecurityType, Terminal
from .parsing import (
    get_paginated_csv_dataframe,
    get_paginated_dataframe_request,
    parse_list,
    parse_trade
)
from .terminal import check_download, launch_terminal
from .utils import _format_date, _format_strike, time_to_ms

_NOT_CONNECTED_MSG = "You must establish a connection first."
_VERSION = '0.9.11'
URL_BASE = "http://127.0.0.1:25510/"

jdk_path = Path.home().joinpath('ThetaData').joinpath('ThetaTerminal') \
    .joinpath('jdk-19.0.1').joinpath('bin')

to_extract = Path.home().joinpath('ThetaData').joinpath('ThetaTerminal')

_thetadata_jar = "ThetaTerminal.jar"


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


class Trade:
    """Trade representing all values provided by the Thetadata stream."""

    def __init__(self):
        """Dummy constructor"""
        self.ms_of_day = 0
        self.sequence = 0
        self.size = 0
        self.condition = TradeCondition.UNDEFINED
        self.price = 0
        self.exchange = None
        self.date = None

    def from_bytes(self, data: bytearray):
        """Deserializes a trade."""
        view = memoryview(data)
        parse_int = lambda d: int.from_bytes(d, "big")
        self.ms_of_day = parse_int(view[0:4])
        self.sequence = parse_int(view[4:8]) & 0xffffffffffffffff
        self.size = parse_int(view[8:12])
        self.condition = TradeCondition.from_code(parse_int(view[12:16]))
        self.price = round(
            parse_int(view[16:20]) * _pt_to_price_mul[parse_int(view[24:28])],
            4)
        self.exchange = Exchange.from_code(parse_int(view[20:24]))
        date_raw = str(parse_int(view[28:32]))
        self.date = date(year=int(date_raw[0:4]), month=int(date_raw[4:6]),
                         day=int(date_raw[6:8]))

    def copy_from(self, other_trade):
        self.ms_of_day = other_trade.ms_of_day
        self.sequence = other_trade.sequence
        self.size = other_trade.size
        self.condition = other_trade.condition
        self.price = other_trade.price
        self.exchange = other_trade.exchange
        self.date = other_trade.date

    def to_string(self) -> str:
        """String representation of a trade."""
        return 'ms_of_day: ' + str(self.ms_of_day) + ' sequence: ' + str(
            self.sequence) + ' size: ' + str(self.size) + \
            ' condition: ' + str(self.condition.name) + ' price: ' + str(
                self.price) + ' exchange: ' + \
            str(self.exchange.value[1]) + ' date: ' + str(self.date)


class OHLCVC:
    """Trade representing all values provided by the Thetadata stream."""

    def __init__(self):
        """Dummy constructor"""
        self.ms_of_day = 0
        self.open = 0
        self.high = 0
        self.low = 0
        self.close = 0
        self.volume = 0
        self.count = 0
        self.date = None

    def from_bytes(self, data: bytearray):
        """Deserializes a trade."""
        view = memoryview(data)
        parse_int = lambda d: int.from_bytes(d, "big")
        self.ms_of_day = parse_int(view[0:4])
        self.open = round(
            parse_int(view[4:8]) * _pt_to_price_mul[parse_int(view[28:32])], 4)
        self.high = round(
            parse_int(view[8:12]) * _pt_to_price_mul[parse_int(view[28:32])],
            4)
        self.low = round(
            parse_int(view[12:16]) * _pt_to_price_mul[parse_int(view[28:32])],
            4)
        self.close = round(
            parse_int(view[16:20]) * _pt_to_price_mul[parse_int(view[28:32])],
            4)
        self.volume = parse_int(view[20:24])
        self.count = parse_int(view[24:28])
        date_raw = str(parse_int(view[32:36]))
        self.date = date(year=int(date_raw[0:4]), month=int(date_raw[4:6]),
                         day=int(date_raw[6:8]))

    def copy_from(self, other_ohlcvc):
        self.ms_of_day = other_ohlcvc.ms_of_day
        self.open = other_ohlcvc.open
        self.high = other_ohlcvc.high
        self.low = other_ohlcvc.low
        self.close = other_ohlcvc.close
        self.volume = other_ohlcvc.volume
        self.count = other_ohlcvc.count
        self.date = other_ohlcvc.date

    def to_string(self) -> str:
        """String representation of a trade."""
        return 'ms_of_day: ' + str(self.ms_of_day) + ' open: ' + str(
            self.open) + ' high: ' + str(self.high) + \
            ' low: ' + str(self.low) + ' close: ' + str(
                self.close) + ' volume: ' + str(self.volume) + \
            ' count: ' + str(self.count) + ' date: ' + str(self.date)


class Quote:
    """Quote representing all values provided by the Thetadata stream."""

    def __init__(self):
        """Dummy constructor"""
        self.ms_of_day = 0
        self.bid_size = 0
        self.bid_exchange = Exchange.OPRA
        self.bid_price = 0
        self.bid_condition = QuoteCondition.UNDEFINED
        self.ask_size = 0
        self.ask_exchange = Exchange.OPRA
        self.ask_price = 0
        self.ask_condition = QuoteCondition.UNDEFINED
        self.date = None

    def from_bytes(self, data: bytes):
        """Deserializes a trade."""
        view = memoryview(data)
        parse_int = lambda d: int.from_bytes(d, "big")
        mult = _pt_to_price_mul[parse_int(view[36:40])]
        self.ms_of_day = parse_int(view[0:4])
        self.bid_size = parse_int(view[4:8])
        self.bid_exchange = Exchange.from_code(parse_int(view[8:12]))
        self.bid_price = round(parse_int(view[12:16]) * mult, 4)
        self.bid_condition = QuoteCondition.from_code(parse_int(view[16:20]))
        self.ask_size = parse_int(view[20:24])
        self.ask_exchange = Exchange.from_code(parse_int(view[24:28]))
        self.ask_price = round(parse_int(view[28:32]) * mult, 4)
        self.ask_condition = QuoteCondition.from_code(parse_int(view[32:36]))
        date_raw = str(parse_int(view[40:44]))
        self.date = date(year=int(date_raw[0:4]), month=int(date_raw[4:6]),
                         day=int(date_raw[6:8]))

    def copy_from(self, other_quote):
        self.ms_of_day = other_quote.ms_of_day
        self.bid_size = other_quote.bid_size
        self.bid_exchange = other_quote.bid_exchange
        self.bid_price = other_quote.bid_price
        self.bid_condition = other_quote.bid_condition
        self.ask_size = other_quote.ask_size
        self.ask_exchange = other_quote.ask_exchange
        self.ask_price = other_quote.ask_price
        self.ask_condition = other_quote.ask_condition
        self.date = other_quote.date

    def to_string(self) -> str:
        """String representation of a quote."""
        return 'ms_of_day: ' + str(self.ms_of_day) + ' bid_size: ' + str(
            self.bid_size) + ' bid_exchange: ' + \
            str(self.bid_exchange.value[1]) + ' bid_price: ' + str(
                self.bid_price) + ' bid_condition: ' + \
            str(self.bid_condition.name) + ' ask_size: ' + str(
                self.ask_size) + ' ask_exchange: ' + \
            str(self.ask_exchange.value[1]) + ' ask_price: ' + str(
                self.ask_price) + ' ask_condition: ' \
            + str(self.ask_condition.name) + ' date: ' + str(self.date)


class OpenInterest:
    """Open Interest"""

    def __init__(self):
        """Dummy constructor"""
        self.open_interest = 0
        self.date = None

    def from_bytes(self, data: bytearray):
        """Deserializes open interest."""
        view = memoryview(data)
        parse_int = lambda d: int.from_bytes(d, "big")
        self.open_interest = parse_int(view[0:4])
        date_raw = str(parse_int(view[4:8]))
        self.date = date(year=int(date_raw[0:4]), month=int(date_raw[4:6]),
                         day=int(date_raw[6:8]))

    def copy_from(self, other_open_interest):
        self.open_interest = other_open_interest.open_interest
        self.date = other_open_interest.date

    def to_string(self) -> str:
        """String representation of open interest."""
        return 'open_interest: ' + str(self.open_interest) + ' date: ' + str(
            self.date)


class Contract:
    """Contract"""

    def __init__(self):
        """Dummy constructor"""
        self.root = ""
        self.exp = None
        self.strike = None
        self.isCall = False
        self.isOption = False

    def from_bytes(self, data: bytes):
        """Deserializes a contract."""
        view = memoryview(data)
        parse_int = lambda d: int.from_bytes(d, "big")
        # parse
        len = parse_int(view[:1])
        root_len = parse_int(view[1:2])
        self.root = data[2:2 + root_len].decode("ascii")

        opt = parse_int(data[root_len + 2: root_len + 3])
        self.isOption = opt == 1
        if not self.isOption:
            return
        date_raw = str(parse_int(view[root_len + 3: root_len + 7]))
        self.exp = date(year=int(date_raw[0:4]), month=int(date_raw[4:6]),
                        day=int(date_raw[6:8]))
        self.isCall = parse_int(view[root_len + 7: root_len + 8]) == 1
        self.strike = parse_int(view[root_len + 9: root_len + 13]) / 1000.0

    def to_string(self) -> str:
        """String representation of open interest."""
        return 'root: ' + self.root + ' isOption: ' + str(
            self.isOption) + ' exp: ' + str(self.exp) + \
            ' strike: ' + str(self.strike) + ' isCall: ' + str(self.isCall)


class StreamMsg:
    """Stream Msg"""

    def __init__(self):
        self.client = None
        self.type = StreamMsgType.ERROR
        self.req_response = None
        self.req_response_id = None
        self.trade = Trade()
        self.ohlcvc = OHLCVC()
        self.quote = Quote()
        self.open_interest = OpenInterest()
        self.contract = Contract()
        self.date = None


def is_security_type(literal: str):
    return literal in get_args(SecurityType)


class ThetaClient:
    """A high-level, blocking client used to fetch market data.

    Instantiating this class runs a java background process, which is
    responsible for the heavy lifting of market data communication.
    Java 11 or higher is required to use this class.

    """

    def __init__(
            self,
            port: int = 25510,
            timeout: Optional[float] = 60,
            launch: bool = True,
            jvm_mem: int = 0,
            username: str = "default",
            passwd: str = "default",
            thetadata_jar = _thetadata_jar,
            auto_update: bool = True,
            use_bundle: bool = True,
            host: str = "127.0.0.1",
            streaming_port: int = 10000,
            move_jar: bool = True,
            stable: bool = True):
        """Construct a client instance to interface with market data.

        If no username and passwd fields are provided, the terminal will
        connect to thetadata servers with free data permissions.

        :param port: The port number specified in the Theta Terminal config,
            which can usually be found under
            %user.home%/ThetaData/ThetaTerminal.
        :param streaming_port: The port number of Theta Terminal Stream server
        :param host: The host name or IP address of Theta Terminal server
        :param timeout: The max number of seconds to wait for a response
            before throwing a TimeoutError
        :param launch: Launches the terminal if true; uses an existing
            external terminal instance if false.
        :param jvm_mem: Any integer provided above zero will force the
            terminal to allocate a maximum amount of memory in GB.
        :param username: Theta Data email. Can be omitted with passwd if
            using free data.
        :param passwd: Theta Data password. Can be omitted with username
            if using free data.
        :param auto_update: If true, this class will automatically download
            the latest terminal version each time this class is instantiated.
            If false, the terminal will use the current jar terminal file.
            If none exists, it will download the latest version.
        :param use_bundle: Will download / use open-jdk-19.0.1 if True and
            the operating system is windows.

        """
        self.host: str = host
        self.port: int = port
        self.username = username,
        self.passwd = passwd
        self.streaming_port: int = streaming_port
        self.timeout = timeout
        self._server: Optional[socket.socket] = None  # None while disconnected
        self._stream_server: Optional[
            socket.socket] = None  # None while disconnected
        self.launch = launch
        self._stream_impl = None
        self._stream_responses = {}
        self._counter_lock = threading.Lock()
        self._stream_req_id = 0
        self._stream_connected = False
        self._thetadata_jar = thetadata_jar,
        self.jvm_mem = jvm_mem,
        self.use_bundle = use_bundle,
        self.move_jar = move_jar,
        self.auto_update = auto_update,
        self.stable = stable,


        logging.info(
            'If you require API support, feel free to join our discord server!'
            'http://discord.thetadata.us')
        if launch:
            terminal.kill_existing_terminal()
            if username == "default" or passwd == "default":
                logging.warning("You are using the free version of Theta Data."
                                " You are currently limited to 20 requests / "
                                "minute.A data subscription can be purchased "
                                "at https://thetadata.net. If you already have"
                                " a ThetaData subscription, specify the "
                                "username and passwd parameters.")

            if check_download(auto_update, stable):
                Thread(target=launch_terminal,
                       args=[username, passwd, use_bundle, jvm_mem,
                             auto_update]).start()
        else:
            print(
                "You are not launching the terminal. This means you should "
                "have an external instance already running.")

    # region  TERMINAL  # TODO add connection manager

    def status(self, service: Optional[Terminal] = 'mdds'):

        url = f"http://{self.host}:{self.port}/v2/system/{service}/status"
        headers = {'Accept': "text/plain"}

        try:
            res = httpx.get(url, headers=headers).raise_for_status()
            logging.info(res.text)
            return True

        except httpx.HTTPError as exc:
            logging.error(f'Could not get {service} status. Error: {exc}')
        except Exception as e:
            logging.error(f'Could not get {service} status. Error: {e}')

    def kill(self, ignore_err=True) -> None:
        """Remotely kill the Terminal process. All subsequent requests will
        time out after this. A new instance of this class must be created.

        """
        try:
            url = f"http://{self.host}:{self.port}/v2/system/terminal/shutdown"
            res = httpx.get(url).raise_for_status()
            logging.info(res.text)
        except Exception as e:
            logging.error(f'Could not close terminal. error: {e}')

    # endregion

    # region LISTING DATA

    def expirations(self, root: str) -> pd.Series:
        """
        Get all options expirations for a provided underlying root.

        :param root:           The root / underlying / ticker / symbol.
        :param host:           The ip address of the server
        :param port:           The port of the server

        :return:               All expirations that ThetaData provides data for.
        :raises ResponseError: If the request failed.
        :raises NoData:        If there is no data available for the request.
        """
        url = f"http://{self.host}:{self.port}/v2/list/expirations"
        params = {"root": root}
        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_list(res=res, dates=True, name="expirations")
        return df

    def roots(self, security_type: SecurityType) -> pd.Series:
        """
        Get all roots for a certain security type.

        :param sec: The type of security.

        :return: All roots / underlyings / tickers / symbols for the security type.
        :raises ResponseError: If the request failed.
        :raises NoData:        If there is no data available for the request.
        """

        if not is_security_type(security_type):
            raise ValueError(f'{security_type} is not a valid {SecurityType}')

        url = f"http://{self.host}:{self.port}/v2/list/roots/{security_type}"
        params = {}
        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_list(res=res, dates=False, name="roots")
        return df

    def option_dates(
            self,
            req: str,
            root: str,
            exp: int,
            right: Optional[OptionRight] = None,
            strike: Optional[int] = None) -> pd.Series:
        """
        Get all dates of data available for a given options contract and request type.

        :param req:            The request type.
        :param sec:             The security type.
        :param root:           The root / underlying / ticker / symbol.
        :param exp:            The expiration date. Must be after the start of `date_range`.
        :param strike:         The strike price in USD.
        :param right:          The right of an options.

        :return:               All dates that Theta Data provides data for given a request.
        :raises ResponseError: If the request failed.
        :raises NoData:        If there is no data available for the request.
        """
        # explicit is better than implicit
        sec = "option"

        url = f"http://{self.host}:{self.port}/v2/list/dates/{sec}/{req}"

        params = {'root': root, 'exp': exp}

        # Add optional params if specific strike and expiration are wanted.
        if right is not None and strike is not None:
            params['right'] = right
            params['strike'] = strike

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_list(res, dates=True, name='dates')
        return df

    def strikes(
            self,
            root: str,
            exp: int) -> pd.Series:
        """
        Get all options strike prices in US tenths of a cent.

        :param root:           The root / underlying / ticker / symbol.
        :param exp:            The expiration date.
        :param date_range:     If specified, this function will return strikes only if they have data for every
                                day in the date range.
        :param host:           The ip address of the server
        :param port:           The port of the server

        :return:               The strike prices on the expiration.
        :raises ResponseError: If the request failed.
        :raises NoData:        If there is no data available for the request.
        """

        params = {"root": root, "exp": exp}
        url = f"http://{self.host}:{self.port}/v2/list/strikes"
        res = httpx.get(url, params=params).raise_for_status().json()
        ser = parse_list(res=res, dates=False, name="strikes")
        ser = ser.divide(1000)
        return ser

    def contracts(
            self,
            root: str,
            start_date: int,
            req: OptionReqType) -> pd.DataFrame:
        """
        Get all options strike prices in US tenths of a cent.

        :param root:           The root / underlying / ticker / symbol.
        :param exp:            The expiration date.
        :param date_range:     If specified, this function will return strikes only if they have data for every
                                day in the date range.
        :param host:           The ip address of the server
        :param port:           The port of the server

        :return:               The strike prices on the expiration.
        :raises ResponseError: If the request failed.
        :raises NoData:        If there is no data available for the request.
        """

        params = {"root": root, "start_date": start_date}
        url = f"http://{self.host}:{self.port}/v2/list/contracts/option/{req}"

        return get_paginated_dataframe_request(url, params)

    # endregion

    # region AT TIME
    def quote_at_time(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: str,
            right: OptionRight,
            root: str,
            strike: float,
            rth: Optional[bool] = True) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/at_time/option/quote"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": time_to_ms(ivl),
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth
        }

        return get_paginated_dataframe_request(url, params)

    def trade_at_time(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: str,
            right: OptionRight,
            root: str,
            strike: float,
            rth: Optional[bool] = True) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/at_time/option/trade"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": time_to_ms(ivl),
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth
        }

        return get_paginated_dataframe_request(url, params)

    # endregion

    # region BULK AT TIME
    # TODO Empty dataframe. Check later.
    def bulk_quote_at_time(
            self,
            start_date: int,
            end_date: int,
            ivl: str,
            root: str,
            exp: Optional[int] = 0,
            rth: Optional[bool] = True) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/bulk_at_time/option/quote"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": time_to_ms(ivl),
            "root": root,
            "rth": rth
        }

        return get_paginated_dataframe_request(url, params)

    # TODO empty dataframe. Check later
    def bulk_trade_at_time(
            self,
            start_date: int,
            end_date: int,
            ivl: str,
            root: str,
            exp: Optional[int] = 0,
            rth: Optional[bool] = True) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/bulk_at_time/option/trade"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": time_to_ms(ivl),
            "root": root,
            "rth": rth
        }

        res = httpx.get(url, params=params,
                        timeout=120).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    # endregion

    # region HISTORICAL DATA
    def historical_eod_report(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            right: OptionRight,
            root: str,
            strike: float) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/hist/option/eod"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "right": right,
            "root": root,
            "strike": strike,

        }

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    def historical_quotes(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/hist/option/quote"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,

        }
        if start_time:
            params["start_time"] = time_to_ms(start_time)
        if start_time:
            params["end_time"] = time_to_ms(end_time)

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    def historical_ohlc(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/hist/option/ohlc"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,

        }
        if start_time:
            params["start_time"] = time_to_ms(start_time)
        if start_time:
            params["end_time"] = time_to_ms(end_time)

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    def historical_open_interest(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/hist/option/open_interest"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,

        }
        if start_time:
            params["start_time"] = time_to_ms(start_time)
        if start_time:
            params["end_time"] = time_to_ms(end_time)

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    def historical_trades(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/hist/option/trade"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,

        }
        if start_time:
            params["start_time"] = time_to_ms(start_time)
        if start_time:
            params["end_time"] = time_to_ms(end_time)

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    def historical_trade_quote(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            exclusive: bool = True) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/hist/option/trade_quote"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,
            "exclusive": exclusive,

        }

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    # endregion

    # region HISTORICAL GREEKS
    def historical_implied_volatility(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            exclusive: bool = True) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/hist/option/implied_volatility")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,
            "exclusive": exclusive,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def historical_greeks(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            exclusive: bool = True) -> pd.DataFrame:
        url = f"http://{self.host}:{self.port}/v2/hist/option/greeks"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,
            "exclusive": exclusive,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def historical_greeks_second_order(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            exclusive: bool = True) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/hist/option/greeks_second_order")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,
            "exclusive": exclusive,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def historical_greeks_third_order(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            exclusive: bool = True) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/hist/option/greeks_third_order")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,
            "exclusive": exclusive,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def historical_all_greeks(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            rth: bool = False,
            exclusive: bool = True) -> pd.DataFrame:
        url = f"http://{self.host}:{self.port}/v2/hist/option/all_greeks"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "rth": rth,
            "exclusive": exclusive,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def historical_trade_greeks(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            perf_boost: True,
            exclusive: bool = True) -> pd.DataFrame:
        url = f"http://{self.host}:{self.port}/v2/hist/option/trade_greeks"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "perf_boost": perf_boost,
            "exclusive": exclusive,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def historical_trade_greeks_second_order(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            perf_boost: True,
            exclusive: bool = True) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/hist/option/trade_greeks_second_order")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "perf_boost": perf_boost,
            "exclusive": exclusive,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def historical_trade_greeks_third_order(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            right: OptionRight,
            root: str,
            strike: float,
            perf_boost: True,
            exclusive: bool = True) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/hist/option/trade_greeks_third_order")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "right": right,
            "root": root,
            "strike": strike,
            "perf_boost": perf_boost,
            "exclusive": exclusive,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    # endregion

    # region BULK HISTORICAL DATA

    def bulk_historical_data_bulk_eod(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            root: str,
            annual_div: Optional[float] = None,
            rate: Optional[Rate] = None,
            rate_value: Optional[float] = None,
            under_price: Optional[float] = None,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/eod")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "root": root,
            "annual_div": annual_div,
            "rate": rate,
            "rate_value": rate_value,
            "under_price": under_price,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_historical_data_bulk_quote(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            root: str,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/quote")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "root": root,
            "start_time": start_time,
            "end_time": end_time,
            "use_csv": True,
        }

        if start_time:
            params["start_time"] = time_to_ms(start_time)
        if start_time:
            params["end_time"] = time_to_ms(end_time)

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_historical_data_bulk_ohlc(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            ivl: int,
            root: str,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/ohlc")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "ivl": ivl,
            "root": root,
            "start_time": start_time,
            "end_time": end_time,
            "use_csv": True,
        }

        if start_time:
            params["start_time"] = time_to_ms(start_time)
        if start_time:
            params["end_time"] = time_to_ms(end_time)

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_historical_data_bulk_open_interest(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            root: str,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/open_interest")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "root": root,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_historical_data_bulk_trade(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            root: str,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/trade")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "root": root,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_historical_data_bulk_trade_quote(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            root: str,
            exclusive: Optional[bool] = True,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/trade_quote")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "root": root,
            "exclusive": exclusive,
            "use_csv": True,
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_historical_data_bulk_eod_greeks(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            root: str,
            annual_div: Optional[float] = None,
            rate: Optional[Rate] = None,
            rate_value: Optional[float] = None,
            under_price: Optional[float] = None,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/eod_greeks")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "root": root,
            "annual_div": annual_div,
            "rate": rate,
            "rate_value": rate_value,
            "under_price": under_price,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_historical_data_bulk_all_greeks(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            root: str,
            ivl: int,
            annual_div: Optional[float] = None,
            rate: Optional[Rate] = None,
            rate_value: Optional[float] = None,
            under_price: Optional[float] = None,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/all_greeks")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "root": root,
            "ivl": ivl,
            "annual_div": annual_div,
            "rate": rate,
            "rate_value": rate_value,
            "under_price": under_price,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    # endregion

    # region BULK HISTORICAL GREEKS

    def bulk_historical_greeks_bulk_trade_greeks(
            self,
            start_date: int,
            end_date: int,
            exp: int,
            root: str,
            ivl: int,
            perf_boost: Optional[bool] = True,
            annual_div: Optional[float] = None,
            rate: Optional[Rate] = None,
            rate_value: Optional[float] = None,
            under_price: Optional[float] = None,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_hist/option/trade_greeks")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "exp": exp,
            "root": root,
            "ivl": ivl,
            "perf_boost": perf_boost,
            "annual_div": annual_div,
            "rate": rate,
            "rate_value": rate_value,
            "under_price": under_price,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    # endregion

    # region SNAPSHOTS

    def snapshots_quotes(
            self,
            exp: int,
            right: OptionRight,
            root: str,
            strike: float) -> pd.DataFrame:
        url = f"http://{self.host}:{self.port}/v2/snapshot/option/quote"

        params = {
            "exp": exp,
            "right": right,
            "root": root,
            "strike": strike,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def snapshots_ohlc(
            self,
            exp: int,
            right: OptionRight,
            root: str,
            strike: float) -> pd.DataFrame:
        url = f"http://{self.host}:{self.port}/v2/snapshot/option/ohlc"

        params = {
            "exp": exp,
            "right": right,
            "root": root,
            "strike": strike,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def snapshots_trade(
            self,
            exp: int,
            right: OptionRight,
            root: str,
            strike: float) -> pd.DataFrame:
        url = f"http://{self.host}:{self.port}/v2/snapshot/option/trade"

        params = {
            "exp": exp,
            "right": right,
            "root": root,
            "strike": strike,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def snapshots_open_interest(
            self,
            exp: int,
            right: OptionRight,
            root: str,
            strike: float) -> pd.DataFrame:
        url = f"http://{self.host}:{self.port}/v2/snapshot/option/open_interest"

        params = {
            "exp": exp,
            "right": right,
            "root": root,
            "strike": strike,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    # endregion

    # region BULK SNAPSHOTS

    def bulk_snapshots_bulk_quotes(
            self,
            exp: int,
            root: str,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_snapshot/option/quote")

        params = {
            "exp": exp,
            "root": root,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_snapshots_bulk_open_interest(
            self,
            exp: int,
            root: str,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_snapshot/option/open_interest")

        params = {
            "exp": exp,
            "root": root,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_snapshots_bulk_ohlc(
            self,
            exp: int,
            root: str,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_snapshot/option/ohlc")

        params = {
            "exp": exp,
            "root": root,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_snapshots_bulk_greeks(
            self,
            exp: int,
            root: str,
            annual_div: Optional[float] = None,
            rate: Optional[Rate] = None,
            rate_value: Optional[float] = None,
            under_price: Optional[float] = None,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_snapshot/option/greeks")

        params = {
            "exp": exp,
            "root": root,
            "annual_div": annual_div,
            "rate": rate,
            "rate_value": rate_value,
            "under_price": under_price,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_snapshots_bulk_greeks_second_order(
            self,
            exp: int,
            root: str,
            annual_div: Optional[float] = None,
            rate: Optional[Rate] = None,
            rate_value: Optional[float] = None,
            under_price: Optional[float] = None,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_snapshot/option/greeks_second_order")

        params = {
            "exp": exp,
            "root": root,
            "annual_div": annual_div,
            "rate": rate,
            "rate_value": rate_value,
            "under_price": under_price,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    def bulk_snapshots_bulk_greeks_third_order(
            self,
            exp: int,
            root: str,
            annual_div: Optional[float] = None,
            rate: Optional[Rate] = None,
            rate_value: Optional[float] = None,
            under_price: Optional[float] = None,
    ) -> pd.DataFrame:
        url = (f"http://{self.host}:{self.port}"
               f"/v2/bulk_snapshot/option/greeks_third_order")

        params = {
            "exp": exp,
            "root": root,
            "annual_div": annual_div,
            "rate": rate,
            "rate_value": rate_value,
            "under_price": under_price,
            "use_csv": True
        }

        return get_paginated_csv_dataframe(
            url, {k: v for k, v in params.items() if v is not None})

    # endregion

    # region #################### STOCKS ########################

    # region HISTORICAL DATA

    def stock_historical_eod_report(
            self,
            start_date: int,
            end_date: int,
            root: str) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/hist/stock/eod"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "root": root,
        }

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    # endregion

    # endregion

    # region ################### INDICES ########################

    def index_eod_report(
            self,
            start_date: int,
            end_date: int,
            root: str) -> pd.DataFrame:

        url = f"http://{self.host}:{self.port}/v2/hist/index/eod"

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "root": root,
        }

        res = httpx.get(url, params=params).raise_for_status().json()
        df = parse_trade(res=res)
        return df

    # endregion

    # region ################# STREAMING ########################## # TODO Have not touched yet
    def connect_stream(self, callback) -> Thread:
        """Initiate a connection with the Theta Terminal Stream server.
        Requests can only be made inside this generator aka the `with client.connect_stream()` block.
        Responses to the provided callback method are recycled, meaning that if you send data received
        in the callback method to another thread, you must create a copy of it first.

        :raises ConnectionRefusedError: If the connection failed.
        :raises TimeoutError: If the timeout is set and has been reached.
        :return: The thread that is responsible for receiving messages.
        """
        for i in range(15):
            try:
                self._stream_server = socket.socket()
                self._stream_server.connect((self.host, self.streaming_port))
                self._stream_server.settimeout(1)
                break
            except ConnectionError:
                if i == 14:
                    raise ConnectionError(
                        'Unable to connect to the local Theta Terminal Stream process. '
                        'Try restarting your system.')
                sleep(1)
        self._stream_server.settimeout(10)
        self._stream_impl = callback
        self._stream_connected = True
        out = Thread(target=self._recv_stream)
        out.start()
        return out

    def close_stream(self):
        self._stream_server.close()

    def req_full_trade_stream_opt(self) -> int:
        """from_bytes
          """
        assert self._stream_server is not None, _NOT_CONNECTED_MSG

        with self._counter_lock:
            req_id = self._stream_req_id
            self._stream_responses[req_id] = None
            self._stream_req_id += 1

        # send request
        hist_msg = f"MSG_CODE={MessageType.STREAM_REQ.value}&sec={SecType.OPTION.value}&req={OptionReqType.TRADE.value}&id={req_id}\n"
        self._stream_server.sendall(hist_msg.encode("utf-8"))
        return req_id

    def req_full_open_interest_stream(self) -> id:
        """from_bytes
          """
        assert self._stream_server is not None, _NOT_CONNECTED_MSG

        with self._counter_lock:
            req_id = self._stream_req_id
            self._stream_responses[req_id] = None
            self._stream_req_id += 1

        # send request
        hist_msg = f"MSG_CODE={MessageType.STREAM_REQ.value}&sec={SecType.OPTION.value}&req={OptionReqType.OPEN_INTEREST.value}&id={req_id}\n"
        self._stream_server.sendall(hist_msg.encode("utf-8"))
        return req_id

    def req_trade_stream_opt(self, root: str, exp: date = 0, strike: float = 0,
                             right: OptionRight = 'C') -> int:
        """from_bytes
          """
        assert self._stream_server is not None, _NOT_CONNECTED_MSG
        # format data
        strike = _format_strike(strike)
        exp_fmt = _format_date(exp)

        with self._counter_lock:
            req_id = self._stream_req_id
            self._stream_responses[req_id] = None
            self._stream_req_id += 1

        # send request
        hist_msg = f"MSG_CODE={MessageType.STREAM_REQ.value}&root={root}&exp={exp_fmt}&strike={strike}&right={right.value}&sec={SecType.OPTION.value}&req={OptionReqType.TRADE.value}&id={req_id}\n"
        self._stream_server.sendall(hist_msg.encode("utf-8"))
        return req_id

    def req_quote_stream_opt(self, root: str, exp: date = 0, strike: float = 0,
                             right: OptionRight = 'C') -> int:
        """from_bytes
          """
        assert self._stream_server is not None, _NOT_CONNECTED_MSG
        # format data
        strike = _format_strike(strike)
        exp_fmt = _format_date(exp)

        with self._counter_lock:
            req_id = self._stream_req_id
            self._stream_responses[req_id] = None
            self._stream_req_id += 1

        # send request
        hist_msg = f"MSG_CODE={MessageType.STREAM_REQ.value}&root={root}&exp={exp_fmt}&strike={strike}&right={right.value}&sec={SecType.OPTION.value}&req={OptionReqType.QUOTE.value}&id={req_id}\n"
        self._stream_server.sendall(hist_msg.encode("utf-8"))
        return req_id

    def remove_full_trade_stream_opt(self) -> int:
        """from_bytes
          """
        assert self._stream_server is not None, _NOT_CONNECTED_MSG

        with self._counter_lock:
            req_id = self._stream_req_id
            self._stream_responses[req_id] = None
            self._stream_req_id += 1

        # send request
        hist_msg = f"MSG_CODE={MessageType.STREAM_REMOVE.value}&sec={SecType.OPTION.value}&req={OptionReqType.TRADE.value}&id={req_id}\n"
        self._stream_server.sendall(hist_msg.encode("utf-8"))
        return req_id

    def remove_full_open_interest_stream(self) -> id:
        """from_bytes
          """
        assert self._stream_server is not None, _NOT_CONNECTED_MSG

        with self._counter_lock:
            req_id = self._stream_req_id
            self._stream_responses[req_id] = None
            self._stream_req_id += 1

        # send request
        hist_msg = f"MSG_CODE={MessageType.STREAM_REMOVE.value}&sec={SecType.OPTION.value}&req={OptionReqType.OPEN_INTEREST.value}&id={req_id}\n"
        self._stream_server.sendall(hist_msg.encode("utf-8"))
        return req_id

    def remove_trade_stream_opt(self, root: str, exp: date = 0,
                                strike: float = 0, right: OptionRight = 'C'):
        """from_bytes
          """
        assert self._stream_server is not None, _NOT_CONNECTED_MSG
        # format data
        strike = _format_strike(strike)
        exp_fmt = _format_date(exp)

        # send request
        hist_msg = f"MSG_CODE={MessageType.STREAM_REMOVE.value}&root={root}&exp={exp_fmt}&strike={strike}&right={right.value}&sec={SecType.OPTION.value}&req={OptionReqType.TRADE.value}&id={-1}\n"
        self._stream_server.sendall(hist_msg.encode("utf-8"))

    def remove_quote_stream_opt(self, root: str, exp: date = 0,
                                strike: float = 0, right: OptionRight = 'C'):
        """from_bytes
          """
        assert self._stream_server is not None, _NOT_CONNECTED_MSG
        # format data
        strike = _format_strike(strike)
        exp_fmt = _format_date(exp)

        with self._counter_lock:
            req_id = self._stream_req_id
            self._stream_responses[req_id] = None
            self._stream_req_id += 1

        # send request
        hist_msg = f"MSG_CODE={MessageType.STREAM_REMOVE.value}&root={root}&exp={exp_fmt}&strike={strike}&right={right.value}&sec={SecType.OPTION.value}&req={OptionReqType.QUOTE.value}&id={-1}\n"
        self._stream_server.sendall(hist_msg.encode("utf-8"))
        return req_id

    def verify(self, req_id: int, timeout: int = 5) -> StreamResponseType:
        tries = 0
        lim = timeout * 100
        while self._stream_responses[req_id] is None:  # This is kind of dumb.
            sleep(.01)
            tries += 1
            if tries >= lim:
                return StreamResponseType.TIMED_OUT

        return self._stream_responses[req_id]

    def _recv_stream(self):
        """from_bytes
          """
        msg = StreamMsg()
        msg.client = self
        parse_int = lambda d: int.from_bytes(d, "big")
        self._stream_server.settimeout(10)
        while self._stream_connected:
            try:
                msg.type = StreamMsgType.from_code(
                    parse_int(self._read_stream(1)[:1]))
                msg.contract.from_bytes(
                    self._read_stream(parse_int(self._read_stream(1)[:1])))
                if msg.type == StreamMsgType.QUOTE:
                    msg.quote.from_bytes(self._read_stream(44))
                elif msg.type == StreamMsgType.TRADE:
                    data = self._read_stream(n_bytes=32)
                    msg.trade.from_bytes(data)
                elif msg.type == StreamMsgType.OHLCVC:
                    data = self._read_stream(n_bytes=36)
                    msg.ohlcvc.from_bytes(data)
                elif msg.type == StreamMsgType.PING:
                    self._read_stream(n_bytes=4)
                elif msg.type == StreamMsgType.OPEN_INTEREST:
                    data = self._read_stream(n_bytes=8)
                    msg.open_interest.from_bytes(data)
                elif msg.type == StreamMsgType.REQ_RESPONSE:
                    msg.req_response_id = parse_int(self._read_stream(4))
                    msg.req_response = StreamResponseType.from_code(
                        parse_int(self._read_stream(4)))
                    self._stream_responses[
                        msg.req_response_id] = msg.req_response
                elif msg.type == StreamMsgType.STOP or msg.type == StreamMsgType.START:
                    msg.date = datetime.strptime(
                        str(parse_int(self._read_stream(4))), "%Y%m%d").date()
                elif msg.type == StreamMsgType.DISCONNECTED or msg.type == StreamMsgType.RECONNECTED:
                    self._read_stream(4)  # Future use.
                else:
                    raise ValueError('undefined msg type: ' + str(msg.type))
            except (ConnectionResetError, OSError) as e:
                msg.type = StreamMsgType.STREAM_DEAD
                self._stream_impl(msg)
                self._stream_connected = False
                return
            except Exception as e:
                msg.type = StreamMsgType.ERROR
                print('Stream error for contract: ' + msg.contract.to_string())
                traceback.print_exc()
            self._stream_impl(msg)

    def _read_stream(self, n_bytes: int) -> bytearray:
        """from_bytes
          """

        buffer = bytearray(self._stream_server.recv(n_bytes))
        total = buffer.__len__()

        while total < n_bytes:
            part = self._stream_server.recv(n_bytes - total)
            if part.__len__() < 0:
                continue
            total += part.__len__()
            buffer.extend(part)
        return buffer

    def _send_ver(self):
        """Sends this API version to the Theta Terminal."""
        ver_msg = f"MSG_CODE={MessageType.HIST.value}&version={_VERSION}\n"
        self._server.sendall(ver_msg.encode("utf-8"))

    def _recv(self, n_bytes: int, progress_bar: bool = False) -> bytearray:
        """Wait for a response from the Terminal.
        :param n_bytes:       The number of bytes to receive.
        :param progress_bar:  Print a progress bar displaying download progress.
        :return:              A response from the Terminal.
        """
        assert self._server is not None, _NOT_CONNECTED_MSG

        # receive body data in parts
        MAX_PART_SIZE = 256  # 4 KiB recommended for most machines

        buffer = bytearray(n_bytes)
        bytes_downloaded = 0

        # tqdm disable=True is slow bc it still calls __new__, which takes nearly 4ms
        range_ = range(0, n_bytes, MAX_PART_SIZE)
        iterable = tqdm(range_, desc="Downloading") if progress_bar else range_

        start = 0
        for i in iterable:
            part_size = min(MAX_PART_SIZE, n_bytes - bytes_downloaded)
            bytes_downloaded += part_size
            part = self._server.recv(part_size)
            if part.__len__() < 0:
                continue
            start += 1
            buffer[i: i + part_size] = part

        assert bytes_downloaded == n_bytes
        return buffer

    # endregion
