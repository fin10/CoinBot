import unittest

from trading_service import TradingService


class TradingServiceTest(unittest.TestCase):

    def testDoTrade(self):
        service = TradingService()
        service.do_trading()
