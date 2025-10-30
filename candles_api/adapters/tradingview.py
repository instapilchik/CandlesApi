"""
Провайдер для получения данных через TradingView WebSocket
"""

import websockets
import json
import random
import string
import re
import asyncio
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import datetime
import pytz
from dataclasses import dataclass

from .base import AbstractDataProvider, MarketType, TimeFrame
from ..core.exceptions import (
    ExchangeNotSupportedException,
    DataFetchException,
    TimeoutException,
)


@dataclass
class TVExchangeConfig:
    """Конфигурация биржи для TradingView"""
    name: str
    spot_prefix: str
    futures_prefix: Optional[str]
    futures_suffix: str

    def get_prefix(self, market_type: str) -> str:
        if market_type in ['futures', 'swap'] and self.futures_prefix:
            return self.futures_prefix
        return self.spot_prefix


class TradingViewProvider(AbstractDataProvider):
    """Провайдер данных через TradingView WebSocket"""

    EXCHANGES: Dict[str, TVExchangeConfig] = {
        'binance': TVExchangeConfig('Binance', 'BINANCE', 'BINANCE', '.P'),
        'bybit': TVExchangeConfig('Bybit', 'BYBIT', 'BYBIT', '.P'),
        'okx': TVExchangeConfig('OKX', 'OKX', 'OKX', '.P'),
        'mexc': TVExchangeConfig('MEXC', 'MEXC', 'MEXC', '.P'),
        'gate': TVExchangeConfig('Gate.io', 'GATEIO', 'GATEIO', '.p'),
        'bitget': TVExchangeConfig('Bitget', 'BITGET', 'BITGET', 'PERP'),
        'kucoin': TVExchangeConfig('KuCoin', 'KUCOIN', 'KUCOIN', 'PERP'),
        'htx': TVExchangeConfig('HTX', 'HUOBI', 'HUOBI', '.P'),
    }

    SUPPORTED_MARKETS = {
        'binance': {'spot': True, 'futures': True},
        'bybit': {'spot': True, 'futures': True},
        'okx': {'spot': True, 'futures': True},
        'mexc': {'spot': True, 'futures': True},
        'gate': {'spot': True, 'futures': True},
        'bitget': {'spot': True, 'futures': True},
        'kucoin': {'spot': True, 'futures': False},
        'htx': {'spot': True, 'futures': True},
    }

    def __init__(self, timeout: int = 10):
        super().__init__(name='tradingview', timeout=timeout)

    async def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: TimeFrame,
        market_type: MarketType = "spot",
        limit: int = 500,
        **kwargs
    ) -> pd.DataFrame:
        self.validate_all(exchange, symbol, timeframe, market_type, limit)

        if not self.supports(exchange, market_type):
            raise ExchangeNotSupportedException(exchange, market_type)

        tv_symbol = self._build_tv_symbol(exchange, symbol, market_type)

        self.logger.info(
            f"Fetching {limit} candles: {tv_symbol} (tf={timeframe})"
        )

        try:
            ws = await self._request_data(tv_symbol, timeframe, limit)
            chart_data = await self._listen(ws, timeout=self.timeout)

            if not chart_data:
                raise DataFetchException(
                    provider='tradingview',
                    exchange=exchange,
                    symbol=symbol,
                    message=f"No data received for {tv_symbol}"
                )

            df = self._extract_ohlcv(chart_data)
            timezone = kwargs.get('timezone', pytz.timezone("UTC"))
            df = self._process_dataframe(df, timezone)
            self._validate_dataframe(df)

            self.logger.info(f"Successfully fetched {len(df)} candles: {tv_symbol}")

            return df

        except TimeoutError as e:
            raise TimeoutException('tradingview', self.timeout, str(e))
        except Exception as e:
            self.logger.error(f"Error fetching {tv_symbol}: {e}")
            raise DataFetchException(
                provider='tradingview',
                exchange=exchange,
                symbol=symbol,
                message=str(e),
                original_error=e
            )

    def supports(self, exchange: str, market_type: MarketType) -> bool:
        exchange_lower = exchange.lower()

        if exchange_lower not in self.EXCHANGES:
            return False

        if exchange_lower not in self.SUPPORTED_MARKETS:
            return False

        return self.SUPPORTED_MARKETS[exchange_lower].get(market_type, False)

    def normalize_symbol(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType
    ) -> str:
        return self._build_tv_symbol(exchange, symbol, market_type)

    def get_supported_exchanges(self) -> List[str]:
        return list(self.EXCHANGES.keys())

    def _build_tv_symbol(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType
    ) -> str:
        exchange_lower = exchange.lower()

        if exchange_lower not in self.EXCHANGES:
            raise ValueError(f"Exchange {exchange} not supported")

        config = self.EXCHANGES[exchange_lower]
        clean_symbol = self._clean_symbol(symbol)
        prefix = config.get_prefix(market_type)

        if market_type == "spot":
            return f"{prefix}:{clean_symbol}"

        return f"{prefix}:{clean_symbol}{config.futures_suffix}"

    @staticmethod
    def _generate_session(session_type: bool) -> str:
        length = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(length))
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
            self.logger.error(f"Error extracting OHLCV: {e}")
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


_provider_instance = None


def get_provider() -> TradingViewProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = TradingViewProvider()
    return _provider_instance
