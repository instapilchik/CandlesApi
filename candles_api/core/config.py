"""
Конфигурация источников данных для бирж
"""

from typing import Dict, Optional, Union, Literal

ProviderType = Literal["tradingview", "ccxt", "hyperliquid"]


class ExchangeConfig:
    """Конфигурация маппинга бирж на провайдеры данных"""

    def __init__(self):
        self.sources = self._get_default_sources()
        self.fallbacks = self._get_default_fallbacks()

    def _get_default_sources(self) -> Dict:
        return {
            'binance': 'tradingview',
            'bybit': 'tradingview',
            'okx': 'tradingview',
            'gate': 'tradingview',
            'mexc': 'tradingview',
            'bitget': 'tradingview',
            'htx': 'tradingview',

            'kucoin': {
                'spot': 'tradingview',
                'futures': 'ccxt'
            },

            'ascendex': 'ccxt',
            'bingx': 'ccxt',

            'hyperliquid': 'hyperliquid',
        }

    def _get_default_fallbacks(self) -> Dict[ProviderType, Optional[ProviderType]]:
        return {
            'tradingview': 'ccxt',
            'ccxt': None,
            'hyperliquid': None
        }

    def get_source(
        self,
        exchange: str,
        market_type: str = 'spot'
    ) -> ProviderType:
        exchange_lower = exchange.lower()
        config = self.sources.get(exchange_lower)

        if config is None:
            return 'tradingview'

        if isinstance(config, str):
            return config

        if isinstance(config, dict):
            return config.get(market_type, 'tradingview')

        return 'tradingview'

    def get_fallback(self, provider: ProviderType) -> Optional[ProviderType]:
        return self.fallbacks.get(provider)

    def set_source(
        self,
        exchange: str,
        provider: Union[ProviderType, Dict[str, ProviderType]],
        market_type: Optional[str] = None
    ) -> None:
        exchange_lower = exchange.lower()

        if market_type is not None:
            if exchange_lower not in self.sources or isinstance(self.sources[exchange_lower], str):
                current = self.sources.get(exchange_lower, 'tradingview')
                self.sources[exchange_lower] = {
                    'spot': current,
                    'futures': current
                }

            if isinstance(provider, str):
                self.sources[exchange_lower][market_type] = provider
        else:
            self.sources[exchange_lower] = provider

    def is_supported(self, exchange: str, provider: ProviderType = None) -> bool:
        exchange_lower = exchange.lower()

        if exchange_lower not in self.sources:
            return False

        if provider is None:
            return True

        source = self.get_source(exchange_lower, 'spot')
        if isinstance(source, str):
            return source == provider

        if isinstance(self.sources[exchange_lower], dict):
            return provider in self.sources[exchange_lower].values()

        return False

    def get_all_exchanges(self, provider: Optional[ProviderType] = None) -> list:
        if provider is None:
            return list(self.sources.keys())

        result = []
        for exchange in self.sources:
            if self.is_supported(exchange, provider):
                result.append(exchange)

        return result

    def to_dict(self) -> Dict:
        return {
            'sources': self.sources,
            'fallbacks': self.fallbacks
        }

    def from_dict(self, data: Dict) -> None:
        if 'sources' in data:
            self.sources = data['sources']
        if 'fallbacks' in data:
            self.fallbacks = data['fallbacks']

    def __repr__(self) -> str:
        return f"<ExchangeConfig(exchanges={len(self.sources)})>"


_config_instance = None


def get_config() -> ExchangeConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = ExchangeConfig()
    return _config_instance
