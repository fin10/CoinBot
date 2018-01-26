import os
import shutil
import unittest

import matplotlib.pyplot as plt

from agent.coin_agent import CoinAgent
from agent.coin_transaction import CoinTransaction
from paths import Paths


class CoinAgentTest(unittest.TestCase):
    __agent = CoinAgent(commission=0.0015)

    def test_train(self):
        if os.path.exists(Paths.MODEL):
            shutil.rmtree(Paths.MODEL)

        transactions = CoinTransaction.get_transactions(Paths.DATA, 'xrp')

        # pivot = int(len(transactions) * 0.8)
        # train_set = transactions[:pivot]
        # test_set = transactions[pivot:]
        # print('Train: {:,}, Test: {:,}'.format(len(train_set), len(test_set)))

        self.__agent.train(transactions, params={
            'r': 0.9,
            'epoch': 20
        })

        # portfolios = self.__agent.evaluate(test_set)
        # plt.plot(portfolios)
        #
        # plt.show()

    def test_evaluate(self):
        portfolios = self.__agent.evaluate('eth')
        plt.plot(portfolios)
        plt.show()

    def test_getting_transactions(self):
        result = CoinTransaction.get_transactions(Paths.DATA, 'eth')
        self.assertGreater(len(result), 0)
