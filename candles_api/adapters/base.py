"""
Базовый класс для провайдеров данных
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Dict, List
import pandas as pd
import re
import logging

from ..core.exceptions import ValidationException

MarketType = Literal["spot", "futures", "swap"]
TimeFrame = Literal["1", "3", "5", "15", "30", "60", "120", "240", "1D", "1W", "1M"]


class AbstractDataProvider(ABC):
    """Базовый класс для всех провайдеров данных (TradingView, CCXT и т.д.)"""

    VALID_TIMEFRAMES = ["1", "3", "5", "15", "30", "60", "120", "240", "1D", "1W", "1M"]
    REQUIRED_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume']

    def __init__(self, name: str = None, timeout: int = 10):
        self.name = name or self.__class__.__name__
        self.timeout = timeout
        self.logger = logging.getLogger(f"candles_api.{self.name}")

    @abstractmethod
    async def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: TimeFrame,
        market_type: MarketType = "spot",
        limit: int = 500,
        **kwargs
    ) -> pd.DataFrame:
        """Получить OHLCV данные для указанных параметров"""
        pass

    @abstractmethod
    def supports(self, exchange: str, market_type: MarketType) -> bool:
        """Проверить поддержку биржи и типа рынка"""
        pass

    @abstractmethod
    def normalize_symbol(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType
    ) -> str:
        """Нормализовать символ для конкретного провайдера"""
        pass

    def get_supported_exchanges(self) -> List[str]:
        """Получить список поддерживаемых бирж"""
        return []

    def get_name(self) -> str:
        return self.name

    def validate_exchange(self, exchange: str) -> None:
        if not exchange or not isinstance(exchange, str):
            raise ValidationException('exchange', exchange, "Exchange must be non-empty string")

        if not re.match(r'^[a-z0-9_]+$', exchange.lower()):
            raise ValidationException(
                'exchange',
                exchange,
                "Exchange name can only contain letters, numbers and underscores"
            )

    def validate_symbol(self, symbol: str) -> None:
        if not symbol or not isinstance(symbol, str):
            raise ValidationException('symbol', symbol, "Symbol must be non-empty string")

        # Remove common separators for validation
        clean_symbol = symbol.replace('/', '').replace('-', '').replace('_', '').replace(':', '')

        if not re.match(r'^[A-Z0-9]+$', clean_symbol.upper()):
            raise ValidationException(
                'symbol',
                symbol,
                "Symbol can only contain letters and numbers"
            )

        if len(clean_symbol) < 3 or len(clean_symbol) > 20:
            raise ValidationException(
                'symbol',
                symbol,
                "Symbol length must be between 3 and 20 characters"
            )

    def validate_timeframe(self, timeframe: str) -> None:
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValidationException(
                'timeframe',
                timeframe,
                f"Invalid timeframe. Valid options: {', '.join(self.VALID_TIMEFRAMES)}"
            )

    def validate_limit(self, limit: int, min_limit: int = 1, max_limit: int = 20000) -> None:
        if not isinstance(limit, int):
            raise ValidationException('limit', limit, "Limit must be an integer")

        if limit < min_limit or limit > max_limit:
            raise ValidationException(
                'limit',
                limit,
                f"Limit must be between {min_limit} and {max_limit}"
            )

    def validate_market_type(self, market_type: str) -> None:
        valid_types = ["spot", "futures", "swap"]
        if market_type not in valid_types:
            raise ValidationException(
                'market_type',
                market_type,
                f"Invalid market type. Valid options: {', '.join(valid_types)}"
            )

    def validate_all(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str,
        limit: int
    ) -> None:
        self.validate_exchange(exchange)
        self.validate_symbol(symbol)
        self.validate_timeframe(timeframe)
        self.validate_market_type(market_type)
        self.validate_limit(limit)

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")

        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return True

    def _clean_symbol(self, symbol: str) -> str:
        return re.sub(r'[^A-Z0-9]', '', symbol.upper())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', timeout={self.timeout})>"
