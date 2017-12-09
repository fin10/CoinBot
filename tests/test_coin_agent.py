import os
import shutil
import unittest

from coin_agent import CoinAgent
from paths import Paths


class CoinAgentTest(unittest.TestCase):
    __agent = CoinAgent(
        commission=0.0015,
        budget=100000,
        num_coin=0,
        coin_value=0,
    )

    def setUp(self):
        if os.path.exists(Paths.MODEL):
            shutil.rmtree(Paths.MODEL)

    def test_train(self):
        self.__agent.train('eth', params={
            'r': 0.9,
            'epoch': 1
        })

    def test_getting_transactions(self):
        result = CoinAgent.get_transactions(Paths.DATA, 'eth')
        self.assertGreater(len(result), 0)
