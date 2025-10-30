"""
API для получения OHLCV данных с криптовалютных бирж
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from .core.manager import get_manager
from .core.exceptions import CandlesAPIException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("candles_api.fastapi")


class OHLCVRequest(BaseModel):
    exchange: str = Field(..., min_length=2, max_length=20, description="Exchange name (e.g., binance)")
    symbol: str = Field(..., min_length=3, max_length=20, description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Timeframe (1, 5, 15, 60, 1D, etc.)")
    market_type: str = Field(default='spot', description="Market type (spot, futures, swap)")
    limit: int = Field(default=500, ge=1, le=20000, description="Number of candles (1-20000)")

    @validator('exchange', 'symbol')
    def lowercase_fields(cls, v):
        return v.lower() if isinstance(v, str) else v

    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ["1", "3", "5", "15", "30", "60", "120", "240", "1D", "1W", "1M"]
        if v not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Valid options: {', '.join(valid_timeframes)}")
        return v

    @validator('market_type')
    def validate_market_type(cls, v):
        valid_types = ["spot", "futures", "swap"]
        if v not in valid_types:
            raise ValueError(f"Invalid market type. Valid options: {', '.join(valid_types)}")
        return v


class CandleData(BaseModel):
    time: int = Field(..., description="Unix timestamp (seconds)")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(default=0.0, description="Volume")


class OHLCVResponse(BaseModel):
    success: bool = Field(..., description="Request status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message (if failed)")


class ExchangeInfo(BaseModel):
    name: str = Field(..., description="Exchange name")
    provider: str = Field(..., description="Data provider (tradingview or ccxt)")
    markets: Dict[str, bool] = Field(..., description="Supported markets")


app = FastAPI(
    title="Candles API",
    description="API для получения OHLCV данных с криптобирж",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_timeframe_seconds(timeframe: str) -> int:
    timeframe_map = {
        '1': 60,           # 1 minute
        '3': 180,          # 3 minutes
        '5': 300,          # 5 minutes
        '15': 900,         # 15 minutes
        '30': 1800,        # 30 minutes
        '60': 3600,        # 1 hour
        '120': 7200,       # 2 hours
        '240': 14400,      # 4 hours
        '1D': 86400,       # 1 day
        '1W': 604800,
        '1M': 2592000,
    }
    return timeframe_map.get(timeframe, 60)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("FastAPI Candles API starting up...")
    logger.info("=" * 60)

    try:
        manager = get_manager()
        exchanges = manager.get_supported_exchanges()
        logger.info(f"Supported exchanges ({len(exchanges)}): {', '.join(exchanges)}")
        logger.info("Candles API ready to serve requests")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI Candles API shutting down...")
    try:
        manager = get_manager()
        await manager.close()
        logger.info("All exchange connections closed")
    except Exception as e:
        logger.error(f"Error closing connections: {e}")

    logger.info("Goodbye!")


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "service": "candles_api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/candles", response_model=OHLCVResponse, tags=["OHLCV"])
async def get_ohlcv_query(
    exchange: str = Query(..., min_length=2, max_length=20, description="Exchange name"),
    symbol: str = Query(..., min_length=3, max_length=20, description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe (1, 5, 15, 60, 1D, etc.)"),
    market_type: str = Query('spot', description="Market type (spot, futures, swap)"),
    limit: int = Query(500, ge=1, le=20000, description="Number of candles"),
    before_timestamp: Optional[int] = Query(None, description="Load candles before this timestamp (Unix seconds)")
):
    try:
        logger.info(
            f"GET /api/candles - {exchange}:{symbol} {timeframe} {market_type} "
            f"limit={limit} before_ts={before_timestamp}"
        )

        manager = get_manager()
        exchange = exchange.lower()
        kwargs = {}

        if before_timestamp:
            kwargs['before'] = before_timestamp
            timeframe_seconds = _get_timeframe_seconds(timeframe)
            since_timestamp = before_timestamp - (limit * timeframe_seconds)
            since_dt = datetime.fromtimestamp(since_timestamp, tz=timezone.utc)
            kwargs['since'] = since_dt
            logger.info(
                f"Loading history: since={since_dt.isoformat()} "
                f"before={datetime.fromtimestamp(before_timestamp, tz=timezone.utc).isoformat()}"
            )

        df = await manager.get_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type,
            limit=limit,
            **kwargs
        )

        candles = []
        for _, row in df.iterrows():
            candle_time = int(row['time'].timestamp())
            if before_timestamp and candle_time > before_timestamp:
                continue

            candles.append({
                'time': candle_time,  # Unix timestamp (seconds)
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0.0
            })

        candles.sort(key=lambda x: x['time'])
        if len(candles) > limit:
            candles = candles[-limit:]

        logger.info(f"Fetched {len(candles)} candles for {exchange}:{symbol}")

        return {
            'success': True,
            'data': {
                'exchange': exchange,
                'symbol': symbol,
                'timeframe': timeframe,
                'market_type': market_type,
                'count': len(candles),
                'candles': candles
            }
        }

    except CandlesAPIException as e:
        logger.error(f"API error: {e}")
        raise HTTPException(
            status_code=e.http_status,
            detail={
                'success': False,
                'error': e.error_code,
                'message': e.message
            }
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': 'internal_error',
                'message': str(e)
            }
        )


@app.post("/api/candles", response_model=OHLCVResponse, tags=["OHLCV"])
async def get_ohlcv_body(request: OHLCVRequest = Body(...)):
    try:
        logger.info(f"POST /api/candles - {request.exchange}:{request.symbol} {request.timeframe} {request.market_type} limit={request.limit}")
        manager = get_manager()

        df = await manager.get_ohlcv(
            exchange=request.exchange,
            symbol=request.symbol,
            timeframe=request.timeframe,
            market_type=request.market_type,
            limit=request.limit
        )

        candles = []
        for _, row in df.iterrows():
            candles.append({
                'time': int(row['time'].timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0.0
            })

        logger.info(f"Fetched {len(candles)} candles for {request.exchange}:{request.symbol}")

        return {
            'success': True,
            'data': {
                'exchange': request.exchange,
                'symbol': request.symbol,
                'timeframe': request.timeframe,
                'market_type': request.market_type,
                'count': len(candles),
                'candles': candles
            }
        }

    except CandlesAPIException as e:
        logger.error(f"API error: {e}")
        raise HTTPException(
            status_code=e.http_status,
            detail={
                'success': False,
                'error': e.error_code,
                'message': e.message
            }
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': 'internal_error',
                'message': str(e)
            }
        )


@app.get("/api/candles/exchanges", tags=["Exchanges"])
async def get_exchanges(
    market_type: Optional[str] = Query(None, description="Filter by market type (spot, futures, swap)")
):
    try:
        logger.info(f"GET /api/candles/exchanges - market_type={market_type}")
        manager = get_manager()
        all_exchanges = manager.get_supported_exchanges()

        exchanges = []
        for exchange in all_exchanges:
            provider = manager.get_provider_for_exchange(exchange, 'spot')
            markets = {
                'spot': manager.is_supported(exchange, 'spot'),
                'futures': manager.is_supported(exchange, 'futures'),
                'swap': manager.is_supported(exchange, 'swap'),
            }

            if market_type:
                if not markets.get(market_type, False):
                    continue

            exchanges.append({
                'name': exchange,
                'provider': provider,
                'markets': markets
            })

        logger.info(f"Returning {len(exchanges)} exchanges")

        return {
            'success': True,
            'data': {
                'count': len(exchanges),
                'exchanges': exchanges
            }
        }

    except Exception as e:
        logger.exception(f"Error getting exchanges: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': 'internal_error',
                'message': str(e)
            }
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'success': False,
            'error': 'internal_error',
            'message': 'An unexpected error occurred'
        }
    )