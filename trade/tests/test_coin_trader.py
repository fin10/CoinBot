import json
import unittest

from logger import logger
from trade.coin_trader import CoinTrader


class CoinTraderTest(unittest.TestCase):

    def test_get_trades(self):
        currency = 'xrp'
        trades = CoinTrader.get_trades(currency)
        logger.debug(trades[-1])

    def test_limit_buy(self):
        price = 10
        currency = 'xrp'
        qty = 1

        response = CoinTrader.limit_buy(price=price, currency=currency, qty=qty)
        logger.debug(response.text)
        self.assertEqual(200, response.status_code)
        self.assertEqual('0', json.loads(response.text)['errorCode'])

    def test_limit_sell(self):
        price = 100000
        currency = 'iota'
        qty = 0.1

        response = CoinTrader.limit_sell(price=price, currency=currency, qty=qty)
        logger.debug(response.text)
        self.assertEqual(200, response.status_code)
        self.assertEqual('0', json.loads(response.text)['errorCode'])
