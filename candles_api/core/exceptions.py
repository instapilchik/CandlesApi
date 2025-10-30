"""
Исключения для API
"""


class CandlesAPIException(Exception):
    """Базовый класс для всех исключений API"""

    http_status = 500
    error_code = 'internal_error'

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {
            'error': self.error_code,
            'message': self.message,
            'details': self.details
        }


class ProviderException(CandlesAPIException):
    """Ошибка провайдера данных"""

    http_status = 502
    error_code = 'provider_error'

    def __init__(self, provider: str, message: str, details: dict = None):
        self.provider = provider
        super().__init__(message, details)
        self.details['provider'] = provider


class ExchangeNotSupportedException(CandlesAPIException):
    """Биржа не поддерживается"""

    http_status = 404
    error_code = 'exchange_not_supported'

    def __init__(self, exchange: str, market_type: str = None):
        self.exchange = exchange
        self.market_type = market_type

        message = f"Exchange '{exchange}' is not supported"
        if market_type:
            message += f" for {market_type} market"

        details = {
            'exchange': exchange,
            'market_type': market_type
        }

        super().__init__(message, details)


class DataFetchException(ProviderException):
    """Ошибка получения данных"""

    error_code = 'data_fetch_error'

    def __init__(
        self,
        provider: str,
        exchange: str,
        symbol: str,
        message: str,
        original_error: Exception = None
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.original_error = original_error

        details = {
            'exchange': exchange,
            'symbol': symbol,
        }

        if original_error:
            details['original_error'] = str(original_error)

        super().__init__(provider, message, details)


class RateLimitException(ProviderException):
    """Превышен лимит запросов"""

    http_status = 429
    error_code = 'rate_limit_exceeded'

    def __init__(
        self,
        provider: str,
        exchange: str,
        retry_after: int = None,
        message: str = None
    ):
        self.exchange = exchange
        self.retry_after = retry_after

        if message is None:
            message = f"Rate limit exceeded for {exchange}"
            if retry_after:
                message += f". Retry after {retry_after} seconds"

        details = {
            'exchange': exchange,
            'retry_after': retry_after
        }

        super().__init__(provider, message, details)


class ValidationException(CandlesAPIException):
    """Ошибка валидации параметров"""

    http_status = 400
    error_code = 'validation_error'

    def __init__(self, field: str, value: any, message: str = None):
        self.field = field
        self.value = value

        if message is None:
            message = f"Validation failed for field '{field}': invalid value '{value}'"

        details = {
            'field': field,
            'value': str(value)
        }

        super().__init__(message, details)


class TimeoutException(ProviderException):
    """Превышен timeout запроса"""

    http_status = 504
    error_code = 'timeout_error'

    def __init__(self, provider: str, timeout: int, message: str = None):
        self.timeout = timeout

        if message is None:
            message = f"Request timeout exceeded ({timeout}s)"

        details = {'timeout': timeout}

        super().__init__(provider, message, details)


class CacheException(CandlesAPIException):
    """Ошибка кеша"""

    http_status = 500
    error_code = 'cache_error'

    def __init__(self, operation: str, message: str, details: dict = None):
        self.operation = operation
        super().__init__(message, details or {})
        self.details['operation'] = operation
