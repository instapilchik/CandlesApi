# Candles API

API для получения OHLCV данных с криптовалютных бирж.

## Описание

Сервис предоставляет единый интерфейс для получения свечных данных с различных бирж через FastAPI. Поддерживает spot и futures рынки.

### Поддерживаемые биржи

- Binance
- Bybit
- OKX
- Gate.io
- MEXC
- Bitget
- HTX (Huobi)
- KuCoin
- Hyperliquid
- AscendEX
- BingX

### Источники данных

Используются три провайдера:
- **TradingView** - основной источник для большинства бирж
- **CCXT** - для бирж без поддержки в TradingView
- **Hyperliquid SDK** - нативный SDK для Hyperliquid

## Установка

```bash
pip install -r requirements_candles.txt
```

## Запуск

```bash
python run_candles_api.py
```

API будет доступно по адресу `http://localhost:8001`

## Использование

### Получение свечей

**GET** `/api/candles`

Параметры:
- `exchange` - название биржи (binance, bybit и т.д.)
- `symbol` - торговая пара (BTCUSDT, ETHUSDT)
- `timeframe` - таймфрейм (1, 5, 15, 60, 1D, 1W)
- `market_type` - тип рынка (spot, futures) - по умолчанию spot
- `limit` - количество свечей (1-20000) - по умолчанию 500

Пример:
```bash
curl "http://localhost:8001/api/candles?exchange=binance&symbol=BTCUSDT&timeframe=60&limit=100"
```

### Список бирж

**GET** `/api/candles/exchanges`

Возвращает список всех поддерживаемых бирж с информацией о провайдерах.

## Структура

```
candles_api/
├── adapters/          # Адаптеры для различных провайдеров
│   ├── base.py        # Базовый класс
│   ├── tradingview.py # TradingView WebSocket
│   ├── ccxt_adapter.py # CCXT провайдер
│   └── hyperliquid_adapter.py # Hyperliquid SDK
├── core/              # Основная логика
│   ├── manager.py     # Менеджер с автоматическим выбором провайдера
│   ├── config.py      # Конфигурация маппинга бирж
│   ├── exceptions.py  # Исключения
│   └── rate_limiter.py # Rate limiting
└── fastapi_app.py     # FastAPI приложение
```

## Таймфреймы

Поддерживаемые интервалы: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1D, 1W, 1M

## API документация

После запуска доступна по адресу `http://localhost:8001/docs`
