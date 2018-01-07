import unittest

from trading_service import TradingService


class TradingServiceTest(unittest.TestCase):

    def testDoTrade(self):
        service = TradingService('xrp', 10000)
        service.do_trading(debug=True)
