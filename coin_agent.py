import json
import os
import random
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

    def __init__(self, commission):
        self.__commission = commission
        self.__budget = 0
        self.__num_coin = 0
        self.__coin_value = 0

    def __get_rewards(self, transactions, actions):
        rewords = []
        portfolios = []
        commission = self.__commission
        budget = self.__budget
        num_coin = self.__num_coin
        coin_value = self.__coin_value

        def get_portfolio():
            return budget + num_coin * coin_value

        for transaction, action in zip(transactions, actions):
            portfolio = get_portfolio()

            coin_value = int(transaction[-1]['price'])
            if action == CoinAgent.Action.BUY:
                budget -= int(coin_value * (1 - commission))
                num_coin += 1
            elif action == CoinAgent.Action.SELL:
                budget += int(coin_value * (1 - commission))
                num_coin -= 1

            portfolios.append(get_portfolio())
            rewords.append((portfolios[-1] - portfolio) * 0.0001)

        return portfolios, rewords

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

    @staticmethod
    def __copy_model(src):
        target = os.path.join(tempfile.gettempdir(), 'target_model')
        if os.path.exists(target):
            shutil.rmtree(target)

        shutil.copytree(src, target)
        print('Copied model to %s.' % target)
        return target

    def train(self, transactions, params):
        r = params['r']
        epoch = params['epoch']

        main_dqn = DQN(Paths.MODEL, len(self.actions))
        copied = self.__copy_model(Paths.MODEL)
        target_dqn = DQN(copied, len(self.actions))

        for n in range(epoch):
            action_dists = main_dqn.predict(transactions)
            action_max_indices = np.argmax(action_dists, axis=1)
            actions = [CoinAgent.Action(index) for index in action_max_indices]

            next_action_dists = target_dqn.predict(transactions[1:])
            next_action_max_values = np.max(next_action_dists, axis=1)

            portfolios, rewards = self.__get_rewards(transactions, actions)

            target_dists = []
            for i in range(len(transactions)):
                target_dist = list(action_dists[i])
                target_dist[action_max_indices[i]] = rewards[i]
                if i < len(transactions) - 1:
                    target_dist[action_max_indices[i]] += r * next_action_max_values[i]
                target_dists.append(target_dist)

            result = main_dqn.train(transactions, target_dists)
            print('[{}] #{}, Loss: {:>8,.4f}, Portfolio: {:>12,.2f}, {}, {}'.format(
                n, result['global_step'], result['loss'],
                portfolios[-1], Counter(actions), Counter([tuple(x) for x in action_dists]).most_common(2)))

            if n > 0 and n % 10 == 0:
                copied = self.__copy_model(Paths.MODEL)
                target_dqn = DQN(copied, len(self.actions))

    def evaluate(self, transactions):
        main_dqn = DQN(Paths.MODEL, len(self.actions))

        action_dists = main_dqn.predict(transactions)
        action_max_indices = np.argmax(action_dists, axis=1)
        actions = [CoinAgent.Action(index) for index in action_max_indices]

        portfolios, _ = self.__get_rewards(transactions, actions)
        print('Portfolio: {:>12,.2f}, {}, {}'.format(
            portfolios[-1], Counter(actions), Counter([tuple(x) for x in action_dists]).most_common(2)))

        return portfolios

    def predict(self, transaction, debug=False):
        if debug:
            return random.choice(CoinAgent.actions), [0, 0, 0]

        main_dqn = DQN(Paths.MODEL, len(self.actions))
        action_dist = main_dqn.predict([transaction])[0]
        action = CoinAgent.Action(action_dist)
        return action, action_dist
