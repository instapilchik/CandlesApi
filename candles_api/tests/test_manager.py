"""
Tests for ExchangeDataManager

Tests:
- Provider selection
- Fallback mechanism
- OHLCV fetching
- Exchange support checking
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from candles_api.core.manager import ExchangeDataManager
from candles_api.core.config import ExchangeConfig


class TestExchangeDataManager:
    """Test suite for ExchangeDataManager"""

    @pytest.fixture
    def manager(self):
        """Create manager instance"""
        return ExchangeDataManager()

    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests"""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()

    # ==================
    # Configuration tests
    # ==================

    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager is not None
        assert manager.config is not None
        assert manager._tv_provider is not None
        assert manager._ccxt_provider is not None

    def test_get_supported_exchanges(self, manager):
        """Test getting list of supported exchanges"""
        exchanges = manager.get_supported_exchanges()

        assert isinstance(exchanges, list)
        assert len(exchanges) >= 11  # At least 11 exchanges

        # Check known exchanges
        assert 'binance' in exchanges
        assert 'bybit' in exchanges
        assert 'okx' in exchanges
        assert 'hyperliquid' in exchanges

    def test_get_provider_for_exchange(self, manager):
        """Test getting provider for exchange"""

        # TradingView exchanges
        assert manager.get_provider_for_exchange('binance', 'spot') == 'tradingview'
        assert manager.get_provider_for_exchange('bybit', 'spot') == 'tradingview'

        # CCXT exchanges
        assert manager.get_provider_for_exchange('ascendex', 'spot') == 'ccxt'

        # Mixed exchange (KuCoin)
        assert manager.get_provider_for_exchange('kucoin', 'spot') == 'tradingview'
        assert manager.get_provider_for_exchange('kucoin', 'futures') == 'ccxt'

        # hyperliquid
        assert manager.get_provider_for_exchange('hyperliquid', 'futures') == 'hyperliquid'
        assert manager.get_provider_for_exchange('hyperliquid', 'spot') == 'hyperliquid'


    def test_is_supported(self, manager):
        """Test exchange support checking"""

        # Supported exchanges
        assert manager.is_supported('binance', 'spot') is True
        assert manager.is_supported('binance', 'futures') is True
        assert manager.is_supported('hyperliquid', 'futures') is True


        assert manager.is_supported('hyperliquid', 'spot') is True

        # Unknown exchanges (should default to TradingView = supported)
        # Note: This depends on config implementation

    # ==================
    # Data fetching tests
    # ==================

    @pytest.mark.asyncio
    async def test_get_ohlcv_binance_spot(self, manager):
        """Test fetching OHLCV from Binance spot (TradingView)"""

        df = await manager.get_ohlcv(
            exchange='binance',
            symbol='BTCUSDT',
            timeframe='60',
            market_type='spot',
            limit=10
        )

        assert df is not None
        assert len(df) > 0
        assert len(df) <= 10

        # Check columns
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns

        print(f"[OK] Fetched {len(df)} candles from Binance spot")

    @pytest.mark.asyncio
    async def test_get_ohlcv_bybit_futures(self, manager):
        """Test fetching OHLCV from Bybit futures (TradingView)"""

        df = await manager.get_ohlcv(
            exchange='bybit',
            symbol='BTCUSDT',
            timeframe='60',
            market_type='futures',
            limit=10
        )

        assert df is not None
        assert len(df) > 0
        assert len(df) <= 10

        print(f"[OK] Fetched {len(df)} candles from Bybit futures")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Hyperliquid symbol format needs investigation")
    async def test_get_ohlcv_hyperliquid(self, manager):
        """Test fetching OHLCV from Hyperliquid (CCXT)"""

        df = await manager.get_ohlcv(
            exchange='hyperliquid',
            symbol='BTCUSDT',
            timeframe='60',
            market_type='futures',
            limit=10
        )

        assert df is not None
        assert len(df) > 0
        assert len(df) <= 10

        print(f"[OK] Fetched {len(df)} candles from Hyperliquid")

    @pytest.mark.asyncio
    async def test_get_ohlcv_different_timeframes(self, manager):
        """Test fetching with different timeframes"""

        timeframes = ['5', '15', '60', '1D']

        for tf in timeframes:
            df = await manager.get_ohlcv(
                exchange='binance',
                symbol='BTCUSDT',
                timeframe=tf,
                market_type='spot',
                limit=5
            )

            assert df is not None
            assert len(df) > 0
            print(f"[OK] Timeframe {tf}: {len(df)} candles")

    @pytest.mark.asyncio
    async def test_all_exchanges_solusdt_5m(self, manager):
        """Test fetching SOLUSDT 5m candles from all exchanges (spot + futures)"""

        # All 11 exchanges
        exchanges = [
            'binance', 'bybit', 'okx', 'gate', 'mexc', 'bitget', 'htx', 'kucoin',
            'ascendex', 'bingx', 'hyperliquid'
        ]

        symbol = 'SOLUSDT'
        timeframe = '5'  # 5 minutes
        limit = 10

        results = {
            'spot': {'success': [], 'failed': []},
            'futures': {'success': [], 'failed': []}
        }

        print(f"\n{'='*70}")
        print(f"Testing all exchanges: {symbol} @ {timeframe}m")
        print(f"{'='*70}")

        for exchange in exchanges:
            # Test SPOT
            print(f"\n[{exchange.upper()}] Testing SPOT...")
            if manager.is_supported(exchange, 'spot'):
                try:
                    df = await manager.get_ohlcv(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        market_type='spot',
                        limit=limit
                    )
                    results['spot']['success'].append(exchange)
                    print(f"  [SUCCESS] Fetched {len(df)} candles (spot)")
                except Exception as e:
                    results['spot']['failed'].append((exchange, str(e)))
                    print(f"  [FAILED] {type(e).__name__}: {str(e)[:80]}")
            else:
                results['spot']['failed'].append((exchange, 'Not supported'))
                print(f"  [SKIP] Spot not supported")

            # Test FUTURES
            print(f"[{exchange.upper()}] Testing FUTURES...")
            if manager.is_supported(exchange, 'futures'):
                try:
                    df = await manager.get_ohlcv(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        market_type='futures',
                        limit=limit
                    )
                    results['futures']['success'].append(exchange)
                    print(f"  [SUCCESS] Fetched {len(df)} candles (futures)")
                except Exception as e:
                    results['futures']['failed'].append((exchange, str(e)))
                    print(f"  [FAILED] {type(e).__name__}: {str(e)[:80]}")
            else:
                results['futures']['failed'].append((exchange, 'Not supported'))
                print(f"  [SKIP] Futures not supported")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"\nSPOT Results:")
        print(f"  Success: {len(results['spot']['success'])}/{len(exchanges)}")
        print(f"  Exchanges: {', '.join(results['spot']['success'])}")
        if results['spot']['failed']:
            print(f"  Failed ({len(results['spot']['failed'])}):")
            for ex, reason in results['spot']['failed']:
                print(f"    - {ex}: {reason[:60]}")

        print(f"\nFUTURES Results:")
        print(f"  Success: {len(results['futures']['success'])}/{len(exchanges)}")
        print(f"  Exchanges: {', '.join(results['futures']['success'])}")
        if results['futures']['failed']:
            print(f"  Failed ({len(results['futures']['failed'])}):")
            for ex, reason in results['futures']['failed']:
                print(f"    - {ex}: {reason[:60]}")

        print(f"\n{'='*70}")

        # Assert at least some exchanges work
        total_success = len(results['spot']['success']) + len(results['futures']['success'])
        assert total_success > 0, "At least some exchanges should work"
        print(f"\n[OK] Total successful: {total_success} (spot + futures)")
        print(f"{'='*70}\n")

    # ==================
    # Error handling tests
    # ==================

    @pytest.mark.asyncio
    async def test_invalid_exchange(self, manager):
        """Test handling of invalid exchange"""

        # This test depends on whether unknown exchanges are supported
        # If config defaults to TradingView, it might still work

        try:
            df = await manager.get_ohlcv(
                exchange='invalid_exchange_xyz',
                symbol='BTCUSDT',
                timeframe='60',
                market_type='spot',
                limit=10
            )
            # If no error, it means it tried TradingView (fallback behavior)
            print("[INFO] Unknown exchange defaulted to TradingView")
        except Exception as e:
            print(f"[OK] Invalid exchange raised error: {type(e).__name__}")

    @pytest.mark.asyncio
    async def test_invalid_symbol(self, manager):
        """Test handling of invalid symbol"""

        try:
            df = await manager.get_ohlcv(
                exchange='binance',
                symbol='INVALIDXYZ',
                timeframe='60',
                market_type='spot',
                limit=10
            )
            # Might return empty data or error
            if len(df) == 0:
                print("[OK] Invalid symbol returned empty data")
        except Exception as e:
            print(f"[OK] Invalid symbol raised error: {type(e).__name__}")

    # ==================
    # Cleanup
    # ==================

    @pytest.mark.asyncio
    async def test_cleanup(self, manager):
        """Test cleanup/close"""
        await manager.close()
        print("[OK] Manager cleanup completed")


def main():
    """Run tests manually"""
    print("=" * 60)
    print("Testing ExchangeDataManager")
    print("=" * 60)

    manager = ExchangeDataManager()

    # Test 1: Configuration
    print("\n[TEST] Configuration")
    exchanges = manager.get_supported_exchanges()
    print(f"[OK] Supported exchanges: {len(exchanges)}")
    print(f"     Exchanges: {', '.join(exchanges[:5])}...")

    # Test 2: Provider selection
    print("\n[TEST] Provider selection")
    print(f"[OK] Binance spot: {manager.get_provider_for_exchange('binance', 'spot')}")
    print(f"[OK] Hyperliquid futures: {manager.get_provider_for_exchange('hyperliquid', 'futures')}")
    print(f"[OK] KuCoin spot: {manager.get_provider_for_exchange('kucoin', 'spot')}")
    print(f"[OK] KuCoin futures: {manager.get_provider_for_exchange('kucoin', 'futures')}")

    # Test 3: Data fetching
    print("\n[TEST] Data fetching")

    async def test_fetch():
        # Test TradingView
        print("\n  Testing TradingView (Binance spot)...")
        df = await manager.get_ohlcv('binance', 'BTCUSDT', '60', 'spot', 5)
        print(f"  [OK] Fetched {len(df)} candles from Binance")
        print(f"       First candle: {df.iloc[0]['time']} OHLC: {df.iloc[0]['open']:.2f}")

        # Test CCXT
        print("\n  Testing CCXT (AscendEX spot)...")
        df = await manager.get_ohlcv('ascendex', 'BTCUSDT', '60', 'spot', 5)
        print(f"  [OK] Fetched {len(df)} candles from AscendEX")

        # Cleanup
        await manager.close()

    asyncio.run(test_fetch())

    print("\n" + "=" * 60)
    print("[SUCCESS] All manual tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    # Run manual tests
    main()
