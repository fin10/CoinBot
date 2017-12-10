import json
import os
import shutil
import tempfile
from collections import Counter
from enum import Enum

import numpy as np

from dqn import DQN
from paths import Paths


class CoinAgent:
    class Action(Enum):
        BUY = 0
        SELL = 1
        NOTHING = 2

    actions = list(Action)

    def __init__(self, commission, budget, num_coin, coin_value):
        self.__commission = commission
        self.__budget = budget
        self.__num_coin = num_coin
        self.__coin_value = coin_value

    def __get_rewards(self, transactions, actions):
        rewords = []
        commission = self.__commission
        budget = self.__budget
        num_coin = self.__num_coin
        coin_value = self.__coin_value

        def get_portfolio():
            return budget + num_coin * coin_value

        for transaction, action in zip(transactions, actions):
            portfolio = get_portfolio()

            coin_value = int(transaction[-1]['price'])
            if action == CoinAgent.Action.BUY and budget > 0:
                bought = (budget / coin_value) * (1 - commission)
                budget -= int(bought * coin_value)
                num_coin += bought
            elif action == CoinAgent.Action.SELL and num_coin > 0:
                budget += int(num_coin * coin_value * (1 - commission))
                num_coin = 0

            new_portfolio = get_portfolio()
            # rewords.append((new_portfolio - portfolio) / portfolio)
            diff = new_portfolio - portfolio
            if diff > 0:
                rewords.append(1.0)
            elif diff < 0:
                rewords.append(-1.0)
            else:
                rewords.append(0.0)

        return get_portfolio(), rewords

    @staticmethod
    def get_transactions(path, currency, max_size=-1):
        transaction_files = []
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                if file.startswith(currency) and file.endswith('.json'):
                    transaction_files.append(os.path.join(root, file))
                if max_size > 0 and max_size == len(transaction_files):
                    break
        transaction_files.sort()

        transactions = []
        for file in transaction_files:
            with open(file, encoding='utf-8') as fp:
                transactions.append(json.load(fp)['orders'])

        return transactions

    def train(self, currency, params):
        r = params['r']
        epoch = params['epoch']

        main_dqn = DQN(Paths.MODEL, len(self.actions))
        target_dqn = DQN(Paths.MODEL, len(self.actions))
        transactions = self.get_transactions(Paths.DATA, currency)

        print('Training starts.')
        for n in range(epoch):
            action_dists = main_dqn.predict(transactions)
            action_max_indices = np.argmax(action_dists, axis=1)
            actions = [CoinAgent.Action(index) for index in action_max_indices]

            next_action_dists = target_dqn.predict(transactions[1:])
            next_action_max_values = np.max(next_action_dists, axis=1)

            portfolio, rewards = self.__get_rewards(transactions, actions)

            for i in range(len(transactions)):
                action_dists[i][action_max_indices[i]] = rewards[i]
                if i < len(transactions) - 1:
                    action_dists[i][action_max_indices[i]] += r * next_action_max_values[i]

            result = main_dqn.train(transactions, action_dists)
            print('[{}] #{}, Loss: {:>8,.4f}, Portfolio: {:>12,.2f}, {}, {}'.format(
                n, result['global_step'], result['loss'],
                portfolio, Counter(actions), Counter([tuple(x) for x in action_dists]).most_common(2)))

            if n > 0 and n % 10 == 0:
                target = os.path.join(tempfile.gettempdir(), 'target_model')
                if os.path.exists(Paths.MODEL):
                    shutil.rmtree(Paths.MODEL)

                shutil.copytree(Paths.MODEL, target)
                target_dqn = DQN(target, len(self.actions))
                print('Copied model to %s.' % target)
