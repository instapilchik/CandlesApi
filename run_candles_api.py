"""
Запуск FastAPI Candles API для отладки в PyCharm
"""

import sys
import os

# Добавляем путь к корню проекта в sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from candles_api.fastapi_app import app

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Starting Candles API in DEBUG mode")
    print("API will be available at: http://127.0.0.1:8001")
    print("Docs: http://127.0.0.1:8001/docs")
    print("=" * 60)

    uvicorn.run(
        app,  # Используем уже импортированное приложение
        host="127.0.0.1",
        port=8001,
        reload=False,  # Отключаем reload для дебаггера
        log_level="info"
    )