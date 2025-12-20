import os

STACK_EXCHANGE_API_BASE = os.getenv('STACK_EXCHANGE_API_BASE', 'https://api.stackexchange.com/2.3')
STACK_EXCHANGE_API_KEY = os.getenv('STACK_EXCHANGE_API_KEY')
STACK_EXCHANGE_RATE_LIMIT = int(os.getenv('STACK_EXCHANGE_RATE_LIMIT', '1000'))


def get_headers(provider: str) -> dict:
    return {
        'User-Agent': 'Resume-Collector/1.0',
        'Accept': 'application/json'
    }
