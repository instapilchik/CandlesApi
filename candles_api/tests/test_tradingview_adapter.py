"""
Тесты для TradingView адаптера
Проверяет получение данных spot и futures для всех поддерживаемых бирж
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, List
import pandas as pd

# Настраиваем путь для импорта
sys.path.insert(0, 'E:/PyProjects/QTS_Web/QTS_lk')
sys.path.insert(0, 'E:/PyProjects/QTS_market_data/market_data')

from candles_api.adapters.tradingview_adapter import TradingViewAdapter

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    """Результат теста для одной биржи"""

    def __init__(self, exchange: str, market_type: str):
        self.exchange = exchange
        self.market_type = market_type
        self.success = False
        self.error = None
        self.candles_count = 0
        self.symbol_tested = None
        self.tv_symbol = None
        self.duration = 0.0
        self.sample_data = None

    def __repr__(self):
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        if self.success:
            return (
                f"{status} | {self.exchange:12s} | {self.market_type:8s} | "
                f"{self.tv_symbol:25s} | {self.candles_count:4d} candles | "
                f"{self.duration:.2f}s"
            )
        else:
            return (
                f"{status} | {self.exchange:12s} | {self.market_type:8s} | "
                f"Error: {self.error}"
            )


class ExchangeTester:
    """Тестировщик для проверки работы с биржами"""

    # Конфигурация тестовых символов для каждой биржи
    TEST_SYMBOLS = {
        'binance': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'bybit': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'okx': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'mexc': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'gate': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'bitget': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'kucoin': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'htx': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'hyperliquid': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'ascendex': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'bingx': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
    }

    def __init__(self, adapter: TradingViewAdapter):
        self.adapter = adapter
        self.results: List[TestResult] = []

    async def test_exchange_market(
        self,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str = "1",
        limit: int = 50
    ) -> TestResult:
        """
        Тестирует получение данных для одной биржи и типа рынка

        Args:
            exchange: Название биржи
            market_type: spot или futures
            symbol: Символ для тестирования
            timeframe: Таймфрейм
            limit: Количество свечей
        """
        result = TestResult(exchange, market_type)
        result.symbol_tested = symbol

        try:
            # Строим TV символ для логирования
            result.tv_symbol = self.adapter.build_tv_symbol(
                exchange, symbol, market_type
            )

            logger.info(
                f"Testing {exchange:12s} {market_type:8s} - {result.tv_symbol}"
            )

            start_time = asyncio.get_event_loop().time()

            # Загружаем данные
            df = await self.adapter.fetch_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                market_type=market_type,
                limit=limit
            )

            result.duration = asyncio.get_event_loop().time() - start_time

            # Проверяем результат
            if df is None or df.empty:
                raise ValueError("DataFrame is empty")

            # Проверяем обязательные колонки
            required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")

            # Проверяем, что есть данные
            if len(df) == 0:
                raise ValueError("No candles returned")

            result.candles_count = len(df)
            result.success = True

            # Сохраняем образец данных (последняя свеча)
            last_candle = df.iloc[-1]
            result.sample_data = {
                'time': last_candle['time'],
                'open': float(last_candle['open']),
                'high': float(last_candle['high']),
                'low': float(last_candle['low']),
                'close': float(last_candle['close']),
                'volume': float(last_candle['volume'])
            }

            logger.info(
                f"✅ {exchange} {market_type}: {len(df)} candles, "
                f"last close: {result.sample_data['close']:.2f}, "
                f"time: {result.duration:.2f}s"
            )

        except Exception as e:
            result.error = str(e)
            logger.error(f"❌ {exchange} {market_type}: {e}")

        return result

    async def test_exchange(
        self,
        exchange: str,
        test_spot: bool = True,
        test_futures: bool = True,
        timeframe: str = "1",
        limit: int = 50
    ) -> List[TestResult]:
        """
        Тестирует одну биржу (spot и/или futures)

        Args:
            exchange: Название биржи
            test_spot: Тестировать spot
            test_futures: Тестировать futures
            timeframe: Таймфрейм для теста
            limit: Количество свечей
        """
        results = []
        test_config = self.TEST_SYMBOLS.get(exchange.lower())

        if not test_config:
            logger.error(f"No test configuration for {exchange}")
            return results

        symbol = test_config['symbol']

        # Тест spot
        if test_spot:
            result = await self.test_exchange_market(
                exchange, "spot", symbol, timeframe, limit
            )
            results.append(result)
            # Небольшая задержка между запросами
            await asyncio.sleep(1)

        # Тест futures
        if test_futures:
            result = await self.test_exchange_market(
                exchange, "futures", symbol, timeframe, limit
            )
            results.append(result)
            await asyncio.sleep(1)

        return results

    async def test_all_exchanges(
        self,
        exchanges: List[str] = None,
        test_spot: bool = True,
        test_futures: bool = True,
        timeframe: str = "1",
        limit: int = 50
    ) -> List[TestResult]:
        """
        Тестирует все биржи или указанный список

        Args:
            exchanges: Список бирж для теста (если None - тестируются все)
            test_spot: Тестировать spot
            test_futures: Тестировать futures
            timeframe: Таймфрейм
            limit: Количество свечей
        """
        if exchanges is None:
            exchanges = list(self.TEST_SYMBOLS.keys())

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Starting tests for {len(exchanges)} exchanges")
        logger.info(f"Spot: {test_spot}, Futures: {test_futures}")
        logger.info(f"Timeframe: {timeframe}, Limit: {limit}")
        logger.info(f"{'=' * 80}\n")

        all_results = []

        for exchange in exchanges:
            logger.info(f"\n--- Testing {exchange.upper()} ---")
            results = await self.test_exchange(
                exchange, test_spot, test_futures, timeframe, limit
            )
            all_results.extend(results)

        self.results = all_results
        return all_results

    def print_summary(self):
        """Выводит сводку по всем тестам"""
        if not self.results:
            logger.warning("No test results to display")
            return

        print("\n" + "=" * 100)
        print("TEST SUMMARY")
        print("=" * 100)

        # Группируем по биржам
        exchanges_results = {}
        for result in self.results:
            if result.exchange not in exchanges_results:
                exchanges_results[result.exchange] = {'spot': None, 'futures': None}
            exchanges_results[result.exchange][result.market_type] = result

        # Выводим результаты
        print(f"\n{'Exchange':<15} {'Spot':<10} {'Futures':<10} {'Details'}")
        print("-" * 100)

        for exchange in sorted(exchanges_results.keys()):
            results = exchanges_results[exchange]
            spot_status = "✅" if results['spot'] and results['spot'].success else "❌"
            futures_status = "✅" if results['futures'] and results['futures'].success else "❌"

            # Детали
            details = []
            if results['spot'] and results['spot'].success:
                details.append(f"Spot: {results['spot'].candles_count} candles")
            if results['futures'] and results['futures'].success:
                details.append(f"Futures: {results['futures'].candles_count} candles")

            print(f"{exchange:<15} {spot_status:<10} {futures_status:<10} {', '.join(details)}")

        # Общая статистика
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful

        print("\n" + "=" * 100)
        print(f"Total tests: {total}")
        print(f"Successful: {successful} ({successful / total * 100:.1f}%)")
        print(f"Failed: {failed} ({failed / total * 100:.1f}%)")
        print("=" * 100)

        # Детали по ошибкам
        if failed > 0:
            print("\n[FAILED TESTS]:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.exchange} ({result.market_type}): {result.error}")

        # Примеры данных
        print("\n[SAMPLE DATA] (last candle from successful tests):")
        for result in self.results:
            if result.success and result.sample_data:
                print(f"\n{result.exchange} {result.market_type} ({result.tv_symbol}):")
                print(f"  Time:   {result.sample_data['time']}")
                print(f"  OHLC:   {result.sample_data['open']:.2f} / "
                      f"{result.sample_data['high']:.2f} / "
                      f"{result.sample_data['low']:.2f} / "
                      f"{result.sample_data['close']:.2f}")
                print(f"  Volume: {result.sample_data['volume']:.2f}")


async def run_full_test():
    """Запускает полный тест всех бирж"""
    adapter = TradingViewAdapter(timeout=5)
    tester = ExchangeTester(adapter)

    # Тестируем все биржи
    results = await tester.test_all_exchanges(
        test_spot=True,
        test_futures=True,
        timeframe="1",  # 1 минута
        limit=50  # 50 свечей
    )

    # Выводим сводку
    tester.print_summary()

    return results


async def run_quick_test():
    """Быстрый тест (только spot, меньше данных)"""
    adapter = TradingViewAdapter(timeout=30)
    tester = ExchangeTester(adapter)

    # Тестируем только spot для быстроты
    exchanges = ['binance', 'bybit', 'okx']  # Только основные биржи

    results = await tester.test_all_exchanges(
        exchanges=exchanges,
        test_spot=True,
        test_futures=False,
        timeframe="5",  # 5 минут
        limit=10  # 10 свечей
    )

    tester.print_summary()
    return results


async def test_specific_exchange(exchange: str, market_type: str = "spot"):
    """Тест конкретной биржи"""
    adapter = TradingViewAdapter(timeout=30)
    tester = ExchangeTester(adapter)

    symbol = tester.TEST_SYMBOLS[exchange.lower()]['symbol']

    result = await tester.test_exchange_market(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe="1",
        limit=100
    )

    print(f"\n{result}")

    if result.success and result.sample_data:
        print("\nSample data:")
        for key, value in result.sample_data.items():
            print(f"  {key}: {value}")

    return result


if __name__ == "__main__":
    import sys

    print("=" * 100)
    print("TradingView Adapter Test Suite")
    print("=" * 100)

    # Выбор режима тестирования
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "quick":
            print("\n[QUICK TEST] Running quick test (main exchanges, spot only)...\n")
            asyncio.run(run_quick_test())

        elif mode == "full":
            print("\n[FULL TEST] Running full test (all exchanges, spot + futures)...\n")
            asyncio.run(run_full_test())

        elif mode in TradingViewAdapter.get_supported_exchanges():
            # Тест конкретной биржи
            market = sys.argv[2] if len(sys.argv) > 2 else "spot"
            print(f"\n[EXCHANGE TEST] Testing {mode.upper()} {market.upper()}...\n")
            asyncio.run(test_specific_exchange(mode, market))

        else:
            print(f"Unknown mode: {mode}")
            print("Usage:")
            print("  python test_tradingview_adapter.py quick          # Quick test")
            print("  python test_tradingview_adapter.py full           # Full test")
            print("  python test_tradingview_adapter.py binance spot   # Specific exchange")

    else:
        # По умолчанию - полный тест
        print("\n[FULL TEST] Running full test (all exchanges, spot + futures)...\n")
        asyncio.run(run_full_test())
