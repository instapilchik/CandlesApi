"""
Провайдер для работы с биржами через библиотеку CCXT
"""

import ccxt.async_support as ccxt
import pandas as pd
import pytz
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from .base import AbstractDataProvider, MarketType, TimeFrame
from ..core.exceptions import (
    ExchangeNotSupportedException,
    DataFetchException,
    TimeoutException,
    RateLimitException,
)
from ..core.rate_limiter import get_limiter


class CCXTProvider(AbstractDataProvider):
    """Провайдер данных через CCXT"""

    TIMEFRAME_MAP = {
        '1': '1m',
        '3': '3m',
        '5': '5m',
        '15': '15m',
        '30': '30m',
        '60': '1h',
        '120': '2h',
        '240': '4h',
        '1D': '1d',
        '1W': '1w',
        '1M': '1M',
    }

    EXCHANGES = {
        'ascendex': {
            'class': ccxt.ascendex,
            'markets': {'spot': True, 'futures': True},
            'options': {
                'defaultType': 'spot',  # or 'swap' for futures
            }
        },
        'bingx': {
            'class': ccxt.bingx,
            'markets': {'spot': True, 'futures': True},
            'options': {
                'defaultType': 'spot',
            }
        },
        'hyperliquid': {
            'class': ccxt.hyperliquid,
            'markets': {'spot': False, 'futures': True},
            'options': {
                'defaultType': 'swap',
            }
        },
        'kucoin': {
            'class': ccxt.kucoin,
            'markets': {'spot': False, 'futures': True},  # Spot via TradingView
            'options': {
                'defaultType': 'swap',
            }
        },
    }

    def __init__(self, timeout: int = 30):
        super().__init__(name='ccxt', timeout=timeout)
        self._exchanges: Dict[str, ccxt.Exchange] = {}
        self._limiter = get_limiter()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        for exchange in self._exchanges.values():
            await exchange.close()
        self._exchanges.clear()

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

        try:
            ex = await self._get_exchange(exchange, market_type)
        except Exception as e:
            raise DataFetchException(
                provider='ccxt',
                exchange=exchange,
                symbol=symbol,
                message=f"Failed to initialize exchange: {e}",
                original_error=e
            )

        ccxt_symbol = self._normalize_symbol_ccxt(symbol, exchange, market_type)
        ccxt_timeframe = self.TIMEFRAME_MAP.get(timeframe, timeframe)

        self.logger.info(
            f"Fetching {limit} candles: {exchange}:{ccxt_symbol} "
            f"(tf={ccxt_timeframe}, type={market_type})"
        )

        try:
            await self._limiter.acquire(exchange)

            since = kwargs.get('since')
            if since and isinstance(since, datetime):
                since = int(since.timestamp() * 1000)

            ohlcv = await ex.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe=ccxt_timeframe,
                since=since,
                limit=limit
            )

            if not ohlcv:
                raise DataFetchException(
                    provider='ccxt',
                    exchange=exchange,
                    symbol=symbol,
                    message=f"No data received for {ccxt_symbol}"
                )

            df = self._ohlcv_to_dataframe(ohlcv)
            timezone = kwargs.get('timezone', pytz.timezone("UTC"))
            df = self._process_dataframe(df, timezone)
            self._validate_dataframe(df)

            self.logger.info(
                f"Successfully fetched {len(df)} candles: "
                f"{exchange}:{ccxt_symbol}"
            )

            return df

        except ccxt.RateLimitExceeded as e:
            raise RateLimitException(
                provider='ccxt',
                exchange=exchange,
                retry_after=60,
                message=str(e)
            )
        except ccxt.RequestTimeout as e:
            raise TimeoutException('ccxt', self.timeout, str(e))
        except ccxt.NetworkError as e:
            raise DataFetchException(
                provider='ccxt',
                exchange=exchange,
                symbol=symbol,
                message=f"Network error: {e}",
                original_error=e
            )
        except Exception as e:
            self.logger.error(f"Error fetching {ccxt_symbol}: {e}")
            raise DataFetchException(
                provider='ccxt',
                exchange=exchange,
                symbol=symbol,
                message=str(e),
                original_error=e
            )

    def supports(self, exchange: str, market_type: MarketType) -> bool:
        exchange_lower = exchange.lower()

        if exchange_lower not in self.EXCHANGES:
            return False

        markets = self.EXCHANGES[exchange_lower]['markets']
        return markets.get(market_type, False)

    def normalize_symbol(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType
    ) -> str:
        return self._normalize_symbol_ccxt(symbol, exchange, market_type)

    def get_supported_exchanges(self) -> List[str]:
        return list(self.EXCHANGES.keys())

    async def _get_exchange(
        self,
        exchange: str,
        market_type: MarketType
    ) -> ccxt.Exchange:
        exchange_lower = exchange.lower()
        cache_key = f"{exchange_lower}_{market_type}"

        if cache_key in self._exchanges:
            return self._exchanges[cache_key]

        if exchange_lower not in self.EXCHANGES:
            raise ValueError(f"Exchange {exchange} not supported")

        config = self.EXCHANGES[exchange_lower]
        exchange_class = config['class']
        options = config['options'].copy()

        if market_type in ['futures', 'swap']:
            options['defaultType'] = 'swap'
        else:
            options['defaultType'] = 'spot'

        ex = exchange_class({
            'enableRateLimit': False,
            'timeout': self.timeout * 1000,
            'options': options,
        })

        await ex.load_markets()
        self._exchanges[cache_key] = ex

        return ex

    def _normalize_symbol_ccxt(
        self,
        symbol: str,
        exchange: str,
        market_type: MarketType
    ) -> str:
        clean = self._clean_symbol(symbol)
        quotes = ['USDT', 'USDC', 'BUSD', 'USD', 'BTC', 'ETH']

        for quote in quotes:
            if clean.endswith(quote):
                base = clean[:-len(quote)]
                if base:
                    result = f"{base}/{quote}"

                    if market_type in ['futures', 'swap']:
                        if exchange.lower() == 'hyperliquid':
                            result = f"{base}/{quote}:USDC"

                    return result

        if len(clean) > 4:
            base = clean[:-4]
            quote = clean[-4:]
            return f"{base}/{quote}"

        return clean

    def _ohlcv_to_dataframe(self, ohlcv: list) -> pd.DataFrame:
        df = pd.DataFrame(
            ohlcv,
            columns=['time', 'open', 'high', 'low', 'close', 'volume']
        )
        df['time'] = df['time'] / 1000
        return df

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        timezone: Optional[pytz.timezone] = None
    ) -> pd.DataFrame:
        if timezone is None:
            timezone = pytz.timezone("UTC")

        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df['time'] = df['time'].dt.tz_convert(timezone)
        df = df.dropna(subset=["open", "high", "low", "close"])
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)

        return df

    def __repr__(self) -> str:
        return f"<CCXTProvider(exchanges={len(self.EXCHANGES)}, active={len(self._exchanges)})>"


_provider_instance = None


def get_provider() -> CCXTProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = CCXTProvider()
    return _provider_instance
