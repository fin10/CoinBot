import argparse
import json
import os
import random
from enum import Enum

import numpy as np
import tensorflow as tf


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# tf.logging.set_verbosity(tf.logging.INFO)


class CoinAgent:
    class Action(Enum):
        NOTHING = 0
        BUY = 1
        SELL = 2

    actions = list(Action)
    num_outputs = len(actions)

    def __init__(self, params):
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.max_length = params['max_length']
        self.estimator = tf.contrib.learn.Estimator(
            model_fn=self.qlearning_model_fn,
            model_dir='./model',
            config=tf.contrib.learn.RunConfig(
                gpu_memory_fraction=params['gpu_memory'],
                save_checkpoints_secs=60,
            ),
            params={
                'hidden_units': params['hidden_units'],
                'learning_rate': params['learning_rate'],
            },
        )

    def input_fn(self, orders, target=[0.0 for _ in range(num_outputs)]):
        if self.max_length < len(orders):
            orders = orders[len(orders) - self.max_length:]

        data = []
        for order in orders:
            price = float(order['price'])
            qty = float(order['qty'])
            data.append([price, qty])

        if self.max_length > len(data):
            for _ in range(self.max_length - len(data)):
                data.insert(0, [0.0, 0.0])

        inputs = {
            'state': tf.constant(data)
        }

        target = tf.constant(target, dtype=tf.float32)

        return inputs, target

    @classmethod
    def qlearning_model_fn(cls, features, target, mode, params):
        state = features['state']
        hidden_units = params['hidden_units']
        learning_rate = params['learning_rate']

        state = tf.nn.l2_normalize(state, 0)

        outputs = tf.contrib.layers.fully_connected(
            inputs=state,
            num_outputs=hidden_units,
        )

        outputs = tf.reshape(outputs, [1, -1])

        q = tf.contrib.layers.fully_connected(
            inputs=outputs,
            num_outputs=cls.num_outputs,
        )

        q = tf.reshape(q, [-1])

        loss = None
        train_op = None
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            loss = tf.reduce_sum(tf.square(target - q))
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=learning_rate,
                optimizer='Adam'
            )

        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions=q,
            loss=loss,
            train_op=train_op
        )

    def select(self, state):
        action_dist = [0.0 for _ in range(self.num_outputs)]

        if os.path.exists('./model'):
            action_dist = list(self.estimator.predict(
                input_fn=lambda: self.input_fn(state)
            ))

        if np.sum(action_dist) == 0 or random.random() > self.epsilon:
            print('random pick %s' % action_dist)
            action_dist = [random.random() for _ in range(self.num_outputs)]

        return action_dist

    def update(self, reward, current_state, current_action_dist, next_action_dist):
        if reward > 0:
            reward = 1.0
        elif reward < 0:
            reward = -1.0

        idx = np.argmax(current_action_dist)
        current_action_dist[idx] = reward + self.gamma * np.max(next_action_dist)
        print('update %s' % current_action_dist)

        self.estimator.fit(
            input_fn=lambda: self.input_fn(current_state, current_action_dist),
            steps=1
        )

    def train(self, trade_files):
        commission = 0.0015
        budget = 10000
        num_coins = 0
        coin_value = 0

        for idx in range(len(trade_files) - 1):
            state = trade_files[idx]
            with open(state) as fp:
                orders = json.load(fp)['orders']

            portfolio = budget + num_coins * coin_value
            action_dist = self.select(orders)
            action = CoinAgent.Action(np.argmax(action_dist))

            coin_value = int(orders[-1]['price'])
            if action == CoinAgent.Action.BUY and budget > 0:
                bought = (budget / coin_value) * (1 - commission)
                budget -= int(bought * coin_value)
                num_coins += bought
            elif action == CoinAgent.Action.SELL and num_coins > 0:
                budget += int(num_coins * coin_value * (1 - commission))
                num_coins = 0

            new_portfolio = budget + num_coins * coin_value
            reward = new_portfolio - portfolio

            next_state = trade_files[idx + 1]
            with open(next_state) as fp:
                next_orders = json.load(fp)['orders']
            next_action_dist = self.select(next_orders)

            self.update(reward, orders, action_dist, next_action_dist)

            print('#%d (%s), Portfolio: %f, Budget: %d, Coin: %f(%d)' %
                  (idx, action, (budget + num_coins * coin_value), budget, num_coins, coin_value))

        return {
            'portfolio': (budget + num_coins * coin_value),
            'budget': budget,
            'num_coins': num_coins,
            'coin_value': coin_value
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('currency')
    parser.add_argument('input')
    args = parser.parse_args()

    files = os.listdir(args.input)
    files = [os.path.join(args.input, file) for file in files
             if file.startswith(args.currency) and file.endswith('.json')]
    files.sort()

    print('# Currency: %s' % args.currency)
    print('# Trades (%d)' % len(files))

    if len(files) == 0:
        print('There is no trades.')
        exit()

    agent = CoinAgent(params={
        'epsilon': 0.9,
        'gamma': 0.01,
        'hidden_units': 50,
        'learning_rate': 0.01,
        'gpu_memory': 0.1,
        'max_length': 12000
    })

    trial = 1
    results = []
    for idx in range(trial):
        result = agent.train(files)
        results.append(result)

    print('- List of Portfolios -')
    for idx, result in enumerate(results, 1):
        print('#%d Portfolio: %f, Budget: %d, Coin: %f(%d)' %
              (idx, result['portfolio'], result['budget'], result['num_coins'], result['coin_value']))
