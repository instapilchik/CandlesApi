"""
Провайдер для получения данных с Hyperliquid через нативный SDK
"""

import asyncio
import time
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pandas as pd
import pytz

from hyperliquid.info import Info
from hyperliquid.utils import constants

from .base import AbstractDataProvider, MarketType, TimeFrame
from ..core.exceptions import (
    ExchangeNotSupportedException,
    DataFetchException,
    TimeoutException,
)


class HyperliquidProvider(AbstractDataProvider):
    """Провайдер данных через официальный Hyperliquid SDK"""

    TIMEFRAME_MAP = {
        '1': '1m',
        '3': '3m',
        '5': '5m',
        '15': '15m',
        '30': '30m',
        '60': '1h',
        '1h': '1h',  # Support both formats
        '120': '2h',
        '2h': '2h',
        '240': '4h',
        '4h': '4h',
        '1D': '1d',
        '1d': '1d',
        '1W': '1w',
        '1w': '1w',
        '1M': '1M',
    }

    MAX_CANDLES = 5000

    VALID_TIMEFRAMES = [
        "1", "3", "5", "15", "30", "60", "120", "240", "1D", "1W", "1M",
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w",
    ]

    WRAPPED_SPOT_TOKENS = {
        'SOL': 'USOL',
        'BTC': 'UBTC',
        'ETH': 'UETH',
    }

    def __init__(self, timeout: int = 30, spot_meta_ttl: int = 3600):
        super().__init__(name='hyperliquid', timeout=timeout)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self._spot_meta_cache: Optional[Dict] = None
        self._spot_meta_timestamp: Optional[float] = None
        self._spot_meta_ttl = spot_meta_ttl
        self._spot_symbol_map: Dict[str, str] = {}
        self.logger.info(
            f"HyperliquidProvider initialized (spot_meta_ttl={spot_meta_ttl}s)"
        )

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

        if exchange.lower() != 'hyperliquid':
            raise ExchangeNotSupportedException(exchange, market_type)

        if limit > self.MAX_CANDLES:
            self.logger.warning(
                f"Limit {limit} exceeds max {self.MAX_CANDLES}, capping to {self.MAX_CANDLES}"
            )
            limit = self.MAX_CANDLES

        # Convert symbol to Hyperliquid format
        coin = await self._normalize_symbol_hyperliquid(symbol, market_type)

        # Convert timeframe
        interval = self._convert_timeframe(timeframe)

        # Calculate time range
        # Support 'before' parameter for historical data (like TradingView/CCXT)
        if 'before' in kwargs and kwargs['before']:
            # before can be timestamp in seconds or milliseconds
            before = kwargs['before']
            if isinstance(before, (int, float)):
                # If before is in seconds (< year 3000), convert to milliseconds
                if before < 10000000000:  # Less than year 2286 in seconds
                    before = int(before * 1000)
                end_time = int(before)
            elif isinstance(before, datetime):
                end_time = int(before.timestamp() * 1000)
            else:
                end_time = int(datetime.now(tz=pytz.UTC).timestamp() * 1000)
        else:
            end_time = int(datetime.now(tz=pytz.UTC).timestamp() * 1000)

        start_time = self._calculate_start_time(end_time, interval, limit)

        self.logger.info(
            f"Fetching {limit} candles: {exchange}:{coin} "
            f"(tf={interval}, type={market_type})"
        )

        try:
            # Fetch data (sync SDK, run in thread pool)
            candles = await asyncio.to_thread(
                self.info.candles_snapshot,  # Method: candles_snapshot (plural!)
                coin,        # name parameter
                interval,    # interval parameter
                start_time,  # startTime parameter
                end_time     # endTime parameter
            )

            if not candles:
                raise DataFetchException(
                    provider='hyperliquid',
                    exchange=exchange,
                    symbol=symbol,
                    message=f"No data received for {coin}"
                )

            # Convert to DataFrame
            df = self._candles_to_dataframe(candles)

            # Process DataFrame
            timezone = kwargs.get('timezone', pytz.UTC)
            df = self._process_dataframe(df, timezone)

            # Validate
            self._validate_dataframe(df)

            self.logger.info(
                f"Successfully fetched {len(df)} candles: {exchange}:{coin}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Error fetching {coin}: {e}")
            raise DataFetchException(
                provider='hyperliquid',
                exchange=exchange,
                symbol=symbol,
                message=str(e),
                original_error=e
            )

    def supports(self, exchange: str, market_type: MarketType) -> bool:
        return (
            exchange.lower() == 'hyperliquid' and
            market_type in ['spot', 'futures']
        )

    def normalize_symbol(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType
    ) -> str:
        return self._clean_symbol(symbol)

    def get_supported_exchanges(self) -> List[str]:
        return ['hyperliquid']

    async def get_available_spot_pairs(self) -> List[str]:
        spot_meta = await self._get_spot_meta()
        pairs = []
        for pair in spot_meta.get('universe', []):
            name = pair.get('name')
            if name:
                pairs.append(name)
        return pairs

    def is_spot_available(self, symbol: str) -> bool:
        if not self._spot_symbol_map:
            return False

        clean = self._clean_symbol(symbol).upper()
        return clean in self._spot_symbol_map

    async def _get_spot_meta(self) -> Dict:
        current_time = time.time()
        if (
            self._spot_meta_cache is not None and
            self._spot_meta_timestamp is not None and
            current_time - self._spot_meta_timestamp < self._spot_meta_ttl
        ):
            cache_age = current_time - self._spot_meta_timestamp
            self.logger.debug(
                f"Using cached spot_meta (age={cache_age:.0f}s, ttl={self._spot_meta_ttl}s)"
            )
            return self._spot_meta_cache

        # Cache expired or missing - fetch fresh data
        self.logger.info("Fetching fresh spot_meta from Hyperliquid")

        try:
            # Run sync method in thread pool
            spot_meta = await asyncio.to_thread(self.info.spot_meta)

            # Update cache
            self._spot_meta_cache = spot_meta
            self._spot_meta_timestamp = current_time

            # Build symbol mapping
            self._build_spot_symbol_map(spot_meta)

            self.logger.info(
                f"Spot metadata cached: {len(spot_meta.get('universe', []))} pairs"
            )

            return spot_meta

        except Exception as e:
            self.logger.error(f"Failed to fetch spot_meta: {e}")
            # If we have stale cache, return it as fallback
            if self._spot_meta_cache is not None:
                self.logger.warning("Using stale cache as fallback")
                return self._spot_meta_cache
            raise

    def _build_spot_symbol_map(self, spot_meta: Dict) -> None:
        """
        Build symbol mapping for fast lookups

        Creates mappings like:
        - HYPE/USDC -> @107
        - HYPEUSDC -> @107
        - hypeusdc -> @107
        - PURR/USDC -> PURR/USDC (special case)
        - SOLUSDC -> @156 (USOL/USDC, wrapped SOL)

        Args:
            spot_meta: Spot metadata from API
        """
        self._spot_symbol_map = {}

        universe = spot_meta.get('universe', [])
        tokens = spot_meta.get('tokens', [])

        for pair in universe:
            name = pair.get('name', '')
            index = pair.get('index')
            token_indices = pair.get('tokens', [])

            if not name or index is None or len(token_indices) < 2:
                continue

            # Get base and quote token names
            base_idx = token_indices[0]
            quote_idx = token_indices[1]

            if base_idx < len(tokens) and quote_idx < len(tokens):
                base_token = tokens[base_idx]['name']
                quote_token = tokens[quote_idx]['name']
            else:
                continue

            # Format: @{index} for most pairs
            coin_format = f"@{index}"

            # Special case for PURR/USDC
            if 'PURR' in name.upper() and 'USDC' in name.upper():
                coin_format = 'PURR/USDC'

            # Add various symbol formats
            # 1. Original name: HYPE/USDC or @1
            self._spot_symbol_map[name] = coin_format

            # 2. Without slash: HYPEUSDC
            no_slash = name.replace('/', '')
            self._spot_symbol_map[no_slash] = coin_format

            # 3. Lowercase: hypeusdc
            self._spot_symbol_map[no_slash.lower()] = coin_format

            # 4. Uppercase: HYPEUSDC
            self._spot_symbol_map[no_slash.upper()] = coin_format

            # 5. Base/Quote format: SOLV/USDC
            pair_name = f"{base_token}/{quote_token}"
            self._spot_symbol_map[pair_name] = coin_format
            self._spot_symbol_map[pair_name.replace('/', '')] = coin_format
            self._spot_symbol_map[pair_name.replace('/', '').upper()] = coin_format

            # 6. Check if this is a wrapped token (USOL = wrapped SOL)
            # Map unwrapped name to wrapped pair
            for unwrapped, wrapped in self.WRAPPED_SPOT_TOKENS.items():
                if base_token == wrapped and quote_token == 'USDC':
                    # USOL/USDC should also be accessible as SOLUSDC
                    unwrapped_pair = f"{unwrapped}USDC"
                    self._spot_symbol_map[unwrapped_pair] = coin_format
                    self._spot_symbol_map[unwrapped_pair.lower()] = coin_format
                    self._spot_symbol_map[f"{unwrapped}/USDC"] = coin_format

                    self.logger.info(
                        f"Mapped wrapped token: {unwrapped}USDC -> {wrapped}/USDC (@{index})"
                    )

        self.logger.debug(
            f"Built spot symbol map: {len(self._spot_symbol_map)} entries"
        )

    def invalidate_spot_cache(self) -> None:
        """
        Manually invalidate spot metadata cache

        Use this if you know spot pairs were added/removed
        and want to force a refresh before TTL expires.
        """
        self.logger.info("Invalidating spot_meta cache")
        self._spot_meta_cache = None
        self._spot_meta_timestamp = None
        self._spot_symbol_map = {}

    # ===================
    # Symbol normalization
    # ===================

    async def _normalize_symbol_hyperliquid(
        self,
        symbol: str,
        market_type: MarketType
    ) -> str:
        """
        Normalize symbol to Hyperliquid format

        Perpetuals: BTCUSDT -> BTC
        Spot: HYPEUSDC -> @107 (via spot_meta lookup)

        Args:
            symbol: Raw symbol (BTCUSDT, HYPEUSDC, etc.)
            market_type: Market type

        Returns:
            str: Hyperliquid coin format

        Examples:
            _normalize_symbol_hyperliquid('BTCUSDT', 'futures') -> 'BTC'
            _normalize_symbol_hyperliquid('HYPEUSDC', 'spot') -> '@107'
            _normalize_symbol_hyperliquid('PURRUSDC', 'spot') -> 'PURR/USDC'
        """
        clean = self._clean_symbol(symbol).upper()

        if market_type == 'futures':
            # Perpetuals: remove USDT/USDC suffix
            coin = clean.replace('USDT', '').replace('USDC', '')
            return coin

        elif market_type == 'spot':
            # Spot: lookup in spot_meta
            spot_meta = await self._get_spot_meta()

            # Try to find in symbol map
            if clean in self._spot_symbol_map:
                return self._spot_symbol_map[clean]

            # Fallback: search in universe
            for pair in spot_meta.get('universe', []):
                pair_name = pair.get('name', '').replace('/', '').upper()
                if clean == pair_name:
                    index = pair.get('index')
                    return f"@{index}"

            # Not found in spot
            coin = clean.replace('USDT', '').replace('USDC', '')

            # Check if this coin has a wrapped version we might have missed
            if coin in self.WRAPPED_SPOT_TOKENS:
                wrapped = self.WRAPPED_SPOT_TOKENS[coin]
                suggestion = f"Note: {coin} spot uses wrapped token {wrapped}/USDC. This should be auto-mapped but wasn't found."
            else:
                suggestion = f"Note: {coin} might be available only as perpetual. Try market_type='futures'."

            raise ValueError(
                f"Spot pair '{symbol}' not found on Hyperliquid. "
                f"{suggestion} "
                f"Available spot pairs: PURR/USDC and {len(spot_meta.get('universe', []))} pairs (@1-@{len(spot_meta.get('universe', []))-1})"
            )

        return clean

    # ===================
    # Private methods
    # ===================

    def _convert_timeframe(self, timeframe: str) -> str:
        """
        Convert our timeframe format to Hyperliquid format

        Args:
            timeframe: Our format (1, 60, 1D, etc.)

        Returns:
            str: Hyperliquid format (1m, 1h, 1d, etc.)
        """
        if timeframe in self.TIMEFRAME_MAP:
            return self.TIMEFRAME_MAP[timeframe]

        # If already in Hyperliquid format, return as-is
        if timeframe in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']:
            return timeframe

        raise ValueError(f"Unsupported timeframe: {timeframe}")

    def _calculate_start_time(
        self,
        end_time: int,
        interval: str,
        limit: int
    ) -> int:
        """
        Calculate start time based on end time, interval, and limit

        Args:
            end_time: End timestamp in milliseconds
            interval: Interval string (1m, 1h, 1d, etc.)
            limit: Number of candles

        Returns:
            int: Start timestamp in milliseconds
        """
        # Parse interval to minutes
        interval_minutes = self._interval_to_minutes(interval)

        # Calculate total time span
        total_minutes = interval_minutes * limit

        # Calculate start time
        start_time = end_time - (total_minutes * 60 * 1000)

        return start_time

    def _interval_to_minutes(self, interval: str) -> int:
        """
        Convert interval string to minutes

        Args:
            interval: Interval (1m, 1h, 1d, etc.)

        Returns:
            int: Minutes
        """
        mapping = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '1d': 1440,
            '1w': 10080,
            '1M': 43200,  # Approximate: 30 days
        }

        return mapping.get(interval, 60)  # Default to 1 hour

    def _candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """
        Convert Hyperliquid candles to DataFrame

        Hyperliquid format:
        [
            {
                "t": 1700000000000,  # Open time (ms)
                "T": 1700003600000,  # Close time (ms)
                "o": "42000.5",      # Open (string!)
                "h": "42500.0",      # High
                "l": "41800.0",      # Low
                "c": "42300.0",      # Close
                "v": "1234.56",      # Volume
                "n": 1523,           # Number of trades
                "s": "BTC",          # Symbol
                "i": "1h"            # Interval
            },
            ...
        ]

        Args:
            candles: List of candle dicts

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        # Extract OHLCV data
        data = []
        for candle in candles:
            data.append({
                'time': candle['t'],  # Use open time
                'open': float(candle['o']),
                'high': float(candle['h']),
                'low': float(candle['l']),
                'close': float(candle['c']),
                'volume': float(candle['v']),
            })

        df = pd.DataFrame(data)

        # Convert timestamp from milliseconds to seconds
        df['time'] = df['time'] / 1000

        return df

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        timezone: Optional[pytz.timezone] = None
    ) -> pd.DataFrame:
        """
        Process DataFrame: add timezone, clean data

        Args:
            df: Raw DataFrame
            timezone: Target timezone (default: UTC)

        Returns:
            Processed DataFrame
        """
        if timezone is None:
            timezone = pytz.UTC

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df['time'] = df['time'].dt.tz_convert(timezone)

        # Remove rows with empty values
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        # Ensure correct types
        df[['open', 'high', 'low', 'close', 'volume']] = df[
            ['open', 'high', 'low', 'close', 'volume']
        ].astype(float)

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        return df

    async def close(self):
        """Close connections (no-op for Hyperliquid SDK)"""
        pass

    def __repr__(self) -> str:
        """String representation"""
        cache_status = 'cached' if self._spot_meta_cache else 'empty'
        return (
            f"<HyperliquidProvider("
            f"spot_cache={cache_status}, "
            f"ttl={self._spot_meta_ttl}s"
            f")>"
        )


# Singleton instance
_provider_instance = None


def get_provider() -> HyperliquidProvider:
    """Get singleton instance of HyperliquidProvider"""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = HyperliquidProvider()
    return _provider_instance
