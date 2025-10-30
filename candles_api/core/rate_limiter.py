"""
Rate limiting для API запросов
"""

import time
import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 300
    delay_between_requests: float = 0.2  # seconds
    burst_limit: int = 10  # max requests in a burst


class SimpleRateLimiter:
    """Простой rate limiter с задержками между запросами"""

    def __init__(self):
        self.logger = logging.getLogger('candles_api.rate_limiter')
        self._last_request_time: Dict[str, float] = {}
        self._delays: Dict[str, float] = {}
        self._default_delay = 0.1
        self._init_default_limits()

    def _init_default_limits(self) -> None:
        default_limits = {
            'ascendex': 0.2,
            'bingx': 0.1,
            'hyperliquid': 0.05,
            'kucoin': 0.3,
            'binance': 0.05,
            'bybit': 0.05,
            'okx': 0.05,
            'gate': 0.05,
            'mexc': 0.05,
            'bitget': 0.05,
            'htx': 0.05,
        }

        self._delays.update(default_limits)

    def set_limit(self, exchange: str, delay: float) -> None:
        self._delays[exchange.lower()] = delay
        self.logger.debug(f"Set rate limit for {exchange}: {delay}s")

    def get_delay(self, exchange: str) -> float:
        return self._delays.get(exchange.lower(), self._default_delay)

    async def acquire(self, exchange: str) -> None:
        exchange_lower = exchange.lower()
        delay = self.get_delay(exchange_lower)
        last_time = self._last_request_time.get(exchange_lower, 0)
        now = time.time()
        elapsed = now - last_time
        wait_time = max(0, delay - elapsed)

        if wait_time > 0:
            self.logger.debug(
                f"Rate limit: waiting {wait_time:.3f}s for {exchange}"
            )
            await asyncio.sleep(wait_time)

        self._last_request_time[exchange_lower] = time.time()

    def reset(self, exchange: Optional[str] = None) -> None:
        if exchange is None:
            self._last_request_time.clear()
            self.logger.debug("Reset all rate limits")
        else:
            exchange_lower = exchange.lower()
            if exchange_lower in self._last_request_time:
                del self._last_request_time[exchange_lower]
            self.logger.debug(f"Reset rate limit for {exchange}")

    def get_stats(self, exchange: str) -> Dict:
        exchange_lower = exchange.lower()
        last_time = self._last_request_time.get(exchange_lower)

        return {
            'exchange': exchange,
            'delay': self.get_delay(exchange_lower),
            'last_request': last_time,
            'time_since_last': time.time() - last_time if last_time else None
        }

    def __repr__(self) -> str:
        return f"<SimpleRateLimiter(exchanges={len(self._delays)})>"


class TokenBucketRateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: int, per: int = 60):
        self.rate = rate
        self.per = per
        self.tokens = float(rate)
        self.last_update = time.time()
        self.logger = logging.getLogger('candles_api.rate_limiter')

    async def acquire(self) -> None:
        while self.tokens < 1:
            now = time.time()
            elapsed = now - self.last_update
            tokens_to_add = elapsed * (self.rate / self.per)

            self.tokens = min(self.rate, self.tokens + tokens_to_add)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.per / self.rate)
                self.logger.debug(f"Token bucket: waiting {wait_time:.3f}s")
                await asyncio.sleep(wait_time)

        self.tokens -= 1

    def __repr__(self) -> str:
        return f"<TokenBucketRateLimiter(rate={self.rate}/{self.per}s, tokens={self.tokens:.2f})>"


class ProxyRotatingRateLimiter:
    """Rate limiter с ротацией прокси"""

    def __init__(self, proxies: list, rate: int):
        self.proxies = proxies
        self.limiters = {
            proxy: TokenBucketRateLimiter(rate, 60)
            for proxy in proxies
        }
        self.logger = logging.getLogger('candles_api.rate_limiter')

    async def acquire(self) -> str:
        for proxy, limiter in self.limiters.items():
            if limiter.tokens >= 1:
                await limiter.acquire()
                return proxy

        proxy = self.proxies[0]
        await self.limiters[proxy].acquire()
        return proxy

    def __repr__(self) -> str:
        return f"<ProxyRotatingRateLimiter(proxies={len(self.proxies)})>"


_limiter_instance = None


def get_limiter() -> SimpleRateLimiter:
    global _limiter_instance
    if _limiter_instance is None:
        _limiter_instance = SimpleRateLimiter()
    return _limiter_instance
