"""
Candles API - Unified API for fetching OHLCV data from multiple exchanges

This package provides:
- Multiple data providers (TradingView, CCXT)
- Automatic fallback mechanisms
- Rate limiting
- Redis caching
- Flask API endpoints
"""

__version__ = '1.0.0'
__author__ = 'QTS Team'

# Will be imported after manager is created to avoid circular imports
_manager_instance = None


def get_manager():
    """
    Get singleton instance of ExchangeDataManager

    Returns:
        ExchangeDataManager: Singleton instance

    Example:
        >>> from candles_api import get_manager
        >>> manager = get_manager()
        >>> df = await manager.fetch_ohlcv('binance', 'BTCUSDT', '1h', 'spot', 500)
    """
    global _manager_instance
    if _manager_instance is None:
        from .core.manager import ExchangeDataManager
        _manager_instance = ExchangeDataManager()
    return _manager_instance


def get_blueprint():
    """
    Get Flask blueprint for Candles API

    Returns:
        Blueprint: Flask blueprint

    Example:
        >>> from candles_api import get_blueprint
        >>> app.register_blueprint(get_blueprint())
    """
    from .api import candles_bp
    return candles_bp

def init_blueprint(app):
    from .api import candles_bp
    app.register_blueprint(get_blueprint())


__all__ = [
    'get_manager',
    'get_blueprint',
]
