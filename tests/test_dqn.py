import os
import shutil
import unittest

from agent.coin_agent import CoinAgent
from agent.coin_transaction import CoinTransaction
from agent.dqn import DQN
from paths import Paths


class CoinAgentTest(unittest.TestCase):

    def setUp(self):
        if os.path.exists(Paths.MODEL):
            shutil.rmtree(Paths.MODEL)

    def test_generate_inputs(self):
        dqn = DQN(Paths.MODEL, len(CoinAgent.actions))
        transactions = CoinTransaction.get_transactions(Paths.DATA, 'eth', max_size=5)

        for result in dqn.generate_input(transactions, []):
            self.assertIsNotNone(result)
