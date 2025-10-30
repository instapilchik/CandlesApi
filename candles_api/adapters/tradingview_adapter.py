"""
Адаптер для получения данных через TradingView
"""

import websockets
import json
import random
import string
import re
import logging
from typing import Optional, Literal, Dict, List
import pandas as pd
import numpy as np
import datetime
import pytz
import asyncio
from dataclasses import dataclass

AvailableTimeFrame = Literal[
    "1", "3", "5", "15", "45", "60", "120", "180", "240",
    "1D", "1W", "1M"
]

MarketType = Literal["spot", "futures", "swap"]


@dataclass
class ExchangeConfig:
    """Конфигурация биржи"""
    name: str
    spot_prefix: str
    futures_prefix: Optional[str]
    futures_suffix: str

    def get_prefix(self, market_type: str) -> str:
        if market_type in ['futures', 'swap'] and self.futures_prefix:
            return self.futures_prefix
        return self.spot_prefix


class TradingViewAdapter:
    """Адаптер для TradingView"""

    EXCHANGES: Dict[str, ExchangeConfig] = {
        'binance': ExchangeConfig(
            name='Binance',
            spot_prefix='BINANCE',
            futures_prefix='BINANCE',
            futures_suffix='.P'
        ),
        'bybit': ExchangeConfig(
            name='Bybit',
            spot_prefix='BYBIT',
            futures_prefix='BYBIT',
            futures_suffix='.P'
        ),
        'okx': ExchangeConfig(
            name='OKX',
            spot_prefix='OKX',
            futures_prefix='OKX',
            futures_suffix='.P'
        ),
        'mexc': ExchangeConfig(
            name='MEXC',
            spot_prefix='MEXC',
            futures_prefix='MEXC',
            futures_suffix='.P'
        ),
        'gate': ExchangeConfig(
            name='Gate.io',
            spot_prefix='GATEIO',
            futures_prefix='GATEIO',
            futures_suffix='.p'
        ),
        'bitget': ExchangeConfig(
            name='Bitget',
            spot_prefix='BITGET',
            futures_prefix='BITGET',
            futures_suffix='PERP'
        ),
        'kucoin': ExchangeConfig(
            name='KuCoin',
            spot_prefix='KUCOIN',
            futures_prefix='KUCOIN',
            futures_suffix='PERP'
        ),
        'htx': ExchangeConfig(
            name='HTX (Huobi)',
            spot_prefix='HUOBI',
            futures_prefix='HUOBI',
            futures_suffix='.P'
        ),
        'hyperliquid': ExchangeConfig(
            name='Hyperliquid',
            spot_prefix='HYPERLIQUID',
            futures_prefix='HYPERLIQUID',
            futures_suffix='PERP'
        ),
        'ascendex': ExchangeConfig(
            name='AscendEX',
            spot_prefix='ASCENDEX',
            futures_prefix='ASCENDEX',
            futures_suffix='PERP'
        ),
        'bingx': ExchangeConfig(
            name='BingX',
            spot_prefix='BINGX',
            futures_prefix='BINGX',
            futures_suffix='PERP'
        ),
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    @classmethod
    def get_supported_exchanges(cls) -> List[str]:
        return list(cls.EXCHANGES.keys())

    def normalize_symbol(self, symbol: str) -> str:
        return re.sub(r'[^A-Z0-9]', '', symbol.upper())

    def build_tv_symbol(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType = "spot"
    ) -> str:
        exchange_lower = exchange.lower()

        if exchange_lower not in self.EXCHANGES:
            raise ValueError(
                f"Exchange '{exchange}' not supported. "
                f"Supported: {', '.join(self.EXCHANGES.keys())}"
            )

        config = self.EXCHANGES[exchange_lower]
        normalized_symbol = self.normalize_symbol(symbol)
        prefix = config.get_prefix(market_type)

        if market_type == "spot":
            return f"{prefix}:{normalized_symbol}"

        return f"{prefix}:{normalized_symbol}{config.futures_suffix}"

    @staticmethod
    def _generate_session(session_type: bool) -> str:
        string_length = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(string_length))
        prefix = "cs" if session_type else "qs"
        return f"{prefix}_{random_string}"

    @staticmethod
    def _prepend_header(sentences: str) -> str:
        return f"~m~{len(sentences)}~m~{sentences}"

    @staticmethod
    def _construct_message(function_name: str, parameters: List) -> str:
        transformed_params = []
        for parameter in parameters:
            if isinstance(parameter, str):
                transformed_params.append(parameter)
            elif isinstance(parameter, dict):
                transformed_params.append(json.dumps(parameter).replace("/", ""))
            else:
                transformed_params.append(parameter)
        return json.dumps(
            {"m": function_name, "p": transformed_params},
            separators=(",", ":")
        )

    def _create_message(self, function_name: str, parameters: List) -> str:
        return self._prepend_header(self._construct_message(function_name, parameters))

    async def _send_message(self, ws, func: str, args: List) -> None:
        message = self._create_message(func, args)
        await ws.send(message)

    async def _request_data(
        self,
        tv_symbol: str,
        time_frame: str,
        look_back_bars: int,
    ) -> any:
        websocket_session = self._generate_session(False)
        chart_session = self._generate_session(True)

        ws = await websockets.connect(
            "wss://data.tradingview.com/socket.io/websocket",
            origin="https://data.tradingview.com"
        )

        resolve_symbol = json.dumps({"symbol": tv_symbol, "adjustment": "splits"})
        chart_session_name = "price"

        await self._send_message(ws, "set_auth_token", ["unauthorized_user_token"])
        await self._send_message(ws, "chart_create_session", [chart_session, ""])
        await self._send_message(ws, "quote_create_session", [websocket_session])
        await self._send_message(
            ws,
            "quote_add_symbols",
            [websocket_session, tv_symbol, {"flags": ["force_permission"]}],
        )
        await self._send_message(
            ws,
            "resolve_symbol",
            [chart_session, "symbol_1", f"={resolve_symbol}"]
        )
        await self._send_message(
            ws,
            "create_series",
            [
                chart_session,
                chart_session_name,
                chart_session_name,
                "symbol_1",
                time_frame,
                look_back_bars,
            ],
        )
        return ws

    async def _listen(self, ws, timeout: int = 30) -> dict:
        chart_data = {}

        try:
            start_time = asyncio.get_event_loop().time()

            while True:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise TimeoutError(f"Data fetch timeout after {timeout}s")

                try:
                    results = await asyncio.wait_for(ws.recv(), timeout=5)

                    pattern = re.compile(r"~m~\d+~m~~h~\d+$")
                    if pattern.match(results):
                        await ws.send(results)
                        continue

                    for r in results.split("~m~"):
                        try:
                            r = json.loads(r)
                            if not isinstance(r, dict):
                                continue

                            message = r.get("m")

                            if message == "timescale_update":
                                p = r.get("p", [])
                                data = [element for element in p if isinstance(element, dict)]
                                if data:
                                    chart_data.update(data[0])
                                    if "price" in chart_data:
                                        return chart_data

                            elif message == "study_error":
                                self.logger.error("TradingView study error")
                                return chart_data

                        except json.JSONDecodeError:
                            continue

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

        finally:
            await ws.close()

        return chart_data

    def _extract_ohlcv(self, chart: dict) -> pd.DataFrame:
        if "price" not in chart or "s" not in chart["price"]:
            raise ValueError("No price data in response")

        try:
            df = pd.DataFrame(
                [st["v"] for st in chart["price"]["s"]],
            ).rename(
                columns={
                    0: "time",
                    1: "open",
                    2: "high",
                    3: "low",
                    4: "close",
                    5: "volume",
                }
            )
            return df
        except Exception as e:
            self.logger.error(f"Error extracting OHLCV data: {e}")
            raise

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        timezone: Optional[pytz.timezone] = None
    ) -> pd.DataFrame:
        if timezone is None:
            timezone = pytz.timezone("UTC")

        df['time'] = df['time'].apply(
            lambda timestamp: datetime.datetime.fromtimestamp(
                timestamp, tz=timezone
            )
        )
        df = df.dropna(subset=["open", "high", "low", "close"])
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(np.float64)

        return df

    async def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: AvailableTimeFrame,
        market_type: MarketType = "spot",
        limit: int = 500,
        timezone: Optional[pytz.timezone] = None,
    ) -> pd.DataFrame:
        tv_symbol = self.build_tv_symbol(exchange, symbol, market_type)

        self.logger.info(
            f"Fetching {limit} candles for {tv_symbol} "
            f"(exchange={exchange}, market={market_type}, tf={timeframe})"
        )

        try:
            ws = await self._request_data(tv_symbol, timeframe, limit)
            chart_data = await self._listen(ws, timeout=self.timeout)

            if not chart_data:
                raise ValueError(f"No data received for {tv_symbol}")

            df = self._extract_ohlcv(chart_data)
            df = self._process_dataframe(df, timezone)

            self.logger.info(f"Successfully fetched {len(df)} candles for {tv_symbol}")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {tv_symbol}: {e}")
            raise

    async def get_latest_candle(
        self,
        exchange: str,
        symbol: str,
        timeframe: AvailableTimeFrame,
        market_type: MarketType = "spot",
        timezone: Optional[pytz.timezone] = None,
    ) -> dict:
        df = await self.fetch_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type,
            limit=1,
            timezone=timezone
        )

        if df.empty:
            raise ValueError("No data received")

        return df.iloc[-1].to_dict()


adapter = TradingViewAdapter()


async def fetch_spot_candles(
    exchange: str,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500
) -> pd.DataFrame:
    return await adapter.fetch_ohlcv(exchange, symbol, timeframe, "spot", limit)


async def fetch_futures_candles(
    exchange: str,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500
) -> pd.DataFrame:
    return await adapter.fetch_ohlcv(exchange, symbol, timeframe, "futures", limit)
