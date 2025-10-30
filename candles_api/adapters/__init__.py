"""Data adapters for different sources"""

from .base import AbstractDataProvider, MarketType, TimeFrame
from .tradingview import TradingViewProvider, get_provider as get_tv_provider
from .ccxt_adapter import CCXTProvider, get_provider as get_ccxt_provider
from .hyperliquid_adapter import HyperliquidProvider, get_provider as get_hyperliquid_provider

# Keep old import for backwards compatibility
from .tradingview_adapter import TradingViewAdapter

__all__ = [
    'AbstractDataProvider',
    'MarketType',
    'TimeFrame',
    'TradingViewProvider',
    'CCXTProvider',
    'HyperliquidProvider',
    'TradingViewAdapter',  # Old version
    'get_tv_provider',
    'get_ccxt_provider',
    'get_hyperliquid_provider',
]
