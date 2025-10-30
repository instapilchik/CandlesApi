"""Core components of Candles API"""

from .exceptions import (
    CandlesAPIException,
    ProviderException,
    ExchangeNotSupportedException,
    DataFetchException,
    RateLimitException,
    ValidationException,
)
from .config import ExchangeConfig
from .rate_limiter import SimpleRateLimiter

__all__ = [
    'CandlesAPIException',
    'ProviderException',
    'ExchangeNotSupportedException',
    'DataFetchException',
    'RateLimitException',
    'ValidationException',
    'ExchangeConfig',
    'SimpleRateLimiter',
]
