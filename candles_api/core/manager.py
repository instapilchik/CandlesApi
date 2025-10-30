"""
Менеджер для работы с биржами и получения OHLCV данных
"""

import logging
from typing import Optional, Dict, List
import pandas as pd
import pytz
from datetime import datetime, timedelta

from ..adapters.base import AbstractDataProvider, MarketType, TimeFrame
from ..adapters.tradingview import get_provider as get_tv_provider
from ..adapters.ccxt_adapter import get_provider as get_ccxt_provider
from ..adapters.hyperliquid_adapter import get_provider as get_hyperliquid_provider
from .config import get_config, ExchangeConfig
from .rate_limiter import get_limiter
from .exceptions import (
    ExchangeNotSupportedException,
    DataFetchException,
    ProviderException,
)


class ExchangeDataManager:
    """Менеджер для получения данных с бирж через разные провайдеры"""

    def __init__(
        self,
        config: Optional[ExchangeConfig] = None,
        enable_cache: bool = False,
        cache_ttl: Optional[Dict[str, int]] = None
    ):
        self.logger = logging.getLogger('candles_api.manager')
        self.config = config or get_config()
        self._tv_provider = get_tv_provider()
        self._ccxt_provider = get_ccxt_provider()
        self._hyperliquid_provider = get_hyperliquid_provider()
        self._limiter = get_limiter()
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl or self._get_default_cache_ttl()
        self._cache = None
        self.logger.info("ExchangeDataManager initialized")

    async def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: TimeFrame,
        market_type: MarketType = "spot",
        limit: int = 500,
        **kwargs
    ) -> pd.DataFrame:
        """Получить OHLCV данные с автоматическим выбором провайдера и fallback"""
        exchange_lower = exchange.lower()
        self.logger.info(
            f"get_ohlcv: {exchange}:{symbol} tf={timeframe} type={market_type} limit={limit}"
        )

        if self.enable_cache:
            cached_data = await self._get_from_cache(
                exchange_lower, symbol, timeframe, market_type, limit
            )
            if cached_data is not None:
                self.logger.info(f"Cache hit: {exchange}:{symbol}")
                return cached_data

        primary_provider_name = self.config.get_source(exchange_lower, market_type)
        primary_provider = self._get_provider(primary_provider_name)

        try:
            df = await primary_provider.fetch_ohlcv(
                exchange=exchange_lower,
                symbol=symbol,
                timeframe=timeframe,
                market_type=market_type,
                limit=limit,
                **kwargs
            )

            if self.enable_cache:
                await self._save_to_cache(
                    exchange_lower, symbol, timeframe, market_type, df
                )

            return df

        except Exception as e:
            self.logger.warning(
                f"Primary provider '{primary_provider_name}' failed: {e}"
            )

            fallback_provider_name = self.config.get_fallback(primary_provider_name)
            if fallback_provider_name:
                self.logger.info(f"Trying fallback: {fallback_provider_name}")
                fallback_provider = self._get_provider(fallback_provider_name)

                if not fallback_provider.supports(exchange_lower, market_type):
                    raise ExchangeNotSupportedException(
                        exchange_lower,
                        market_type
                    )

                try:
                    df = await fallback_provider.fetch_ohlcv(
                        exchange=exchange_lower,
                        symbol=symbol,
                        timeframe=timeframe,
                        market_type=market_type,
                        limit=limit,
                        **kwargs
                    )

                    if self.enable_cache:
                        await self._save_to_cache(
                            exchange_lower, symbol, timeframe, market_type, df
                        )

                    return df

                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback provider '{fallback_provider_name}' also failed: {fallback_error}"
                    )
                    raise DataFetchException(
                        provider=fallback_provider_name,
                        exchange=exchange_lower,
                        symbol=symbol,
                        message=f"Both providers failed. Primary: {e}. Fallback: {fallback_error}",
                        original_error=fallback_error
                    )
            else:
                raise

    async def get_available_symbols(
        self,
        exchange: str,
        market_type: MarketType = "spot"
    ) -> List[str]:
        """Получить список доступных символов для биржи"""
        exchange_lower = exchange.lower()
        provider_name = self.config.get_source(exchange_lower, market_type)
        provider = self._get_provider(provider_name)

        if provider_name == 'ccxt':
            try:
                from ..adapters.ccxt_adapter import CCXTProvider
                ccxt_provider: CCXTProvider = provider
                ex = await ccxt_provider._get_exchange(exchange_lower, market_type)
                return list(ex.symbols)
            except Exception as e:
                self.logger.error(f"Failed to get symbols: {e}")
                return []

        return []

    def is_supported(
        self,
        exchange: str,
        market_type: MarketType = "spot"
    ) -> bool:
        """Проверить поддержку биржи"""
        exchange_lower = exchange.lower()
        provider_name = self.config.get_source(exchange_lower, market_type)
        provider = self._get_provider(provider_name)
        is_primary_supported = provider.supports(exchange_lower, market_type)

        if is_primary_supported:
            return True

        fallback_name = self.config.get_fallback(provider_name)
        if fallback_name:
            fallback_provider = self._get_provider(fallback_name)
            return fallback_provider.supports(exchange_lower, market_type)

        return False

    def get_supported_exchanges(self) -> List[str]:
        """Получить список всех поддерживаемых бирж"""
        return self.config.get_all_exchanges()

    def get_provider_for_exchange(
        self,
        exchange: str,
        market_type: MarketType = "spot"
    ) -> str:
        """Получить имя провайдера для биржи"""
        return self.config.get_source(exchange.lower(), market_type)

    def _get_provider(self, provider_name: str) -> AbstractDataProvider:
        if provider_name == 'tradingview':
            return self._tv_provider
        elif provider_name == 'ccxt':
            return self._ccxt_provider
        elif provider_name == 'hyperliquid':
            return self._hyperliquid_provider
        else:
            raise ProviderException(
                provider_name,
                f"Unknown provider: {provider_name}"
            )

    def _get_default_cache_ttl(self) -> Dict[str, int]:
        return {
            'closed_candles': 86400,
            'recent_candles': 300,
            'current_candle': 60,
        }

    async def _get_from_cache(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str,
        limit: int
    ) -> Optional[pd.DataFrame]:
        # TODO: реализовать кеширование
        return None

    async def _save_to_cache(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        market_type: str,
        df: pd.DataFrame
    ) -> None:
        # TODO: реализовать кеширование
        pass

    async def close(self):
        await self._ccxt_provider.close()
        await self._hyperliquid_provider.close()

    def __repr__(self) -> str:
        return (
            f"<ExchangeDataManager("
            f"exchanges={len(self.config.get_all_exchanges())}, "
            f"cache={'enabled' if self.enable_cache else 'disabled'}"
            f")>"
        )


_manager_instance = None


def get_manager() -> ExchangeDataManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ExchangeDataManager()
    return _manager_instance
