"""Check for gaps in candles data"""
import requests
from datetime import datetime

url = "http://localhost:8001/api/candles"
params = {
    'exchange': 'kucoin',
    'symbol': 'BTCUSDT',
    'timeframe': '5',
    'market_type': 'futures',
    'limit': 2500,
    'before_timestamp': 1760875200
}

print(f"Fetching candles from {url}")
print(f"Parameters: {params}\n")

response = requests.get(url, params=params)
data = response.json()

candles = data['data']['candles']

print(f"Total candles: {len(candles)}")
if len(candles) > 0:
    print(f"First: {datetime.fromtimestamp(candles[0]['time'])}")
    print(f"Last: {datetime.fromtimestamp(candles[-1]['time'])}")

print('\nChecking for gaps (expecting 5 min = 300 sec intervals):')
gaps_found = 0
for i in range(1, len(candles)):
    prev_time = candles[i-1]['time']
    curr_time = candles[i]['time']
    diff = curr_time - prev_time

    if diff != 300:  # Should be 300 seconds (5 min)
        gaps_found += 1
        gap_hours = diff / 3600
        gap_mins = diff / 60
        print(f'\n[GAP #{gaps_found}] at index {i}:')
        print(f'  Prev: {datetime.fromtimestamp(prev_time)}')
        print(f'  Curr: {datetime.fromtimestamp(curr_time)}')
        print(f'  Gap: {diff} seconds = {gap_mins:.1f} minutes = {gap_hours:.2f} hours')

if gaps_found == 0:
    print('\nNo gaps found - data is continuous!')
else:
    print(f'\nTotal gaps found: {gaps_found}')
