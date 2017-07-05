import argparse
import json
import os
import random
from enum import Enum

import numpy as np
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.logging.set_verbosity(tf.logging.INFO)


class CoinAgent:
    class Action(Enum):
        NOTHING = 0
        BUY = 1
        SELL = 2

    def __init__(self, params):
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.max_length = params['max_length']
        self.estimator = tf.contrib.learn.Estimator(
            model_fn=self.qlearning_model_fn,
            model_dir='./model',
            config=tf.contrib.learn.RunConfig(
                gpu_memory_fraction=params['gpu_memory'],
                save_checkpoints_secs=30,
            ),
            params={
                'hidden_units': params['hidden_units'],
                'learning_rate': params['learning_rate'],
            },
        )

    def input_fn(self, orders, target=[0.0 for _ in range(len(list(Action)))]):
        if self.max_length < len(orders):
            orders = orders[len(orders) - self.max_length:]

        data = []
        first_time = float(orders[0]['timestamp'])

        for order in orders:
            price = float(order['price'])
            qty = float(order['qty'])
            timestamp = float(order['timestamp']) - first_time
            data.append([price, qty, timestamp])

        if self.max_length > len(data):
            for _ in range(self.max_length - len(data)):
                data.append([0.0, 0.0, 0.0])

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
        num_outputs = len(list(cls.Action))

        state = tf.stack([state])
        target = tf.stack([target])

        outputs = tf.contrib.layers.fully_connected(
            inputs=state,
            num_outputs=hidden_units,
        )

        outputs = tf.reshape(outputs, [1, -1])

        q = tf.contrib.layers.fully_connected(
            inputs=outputs,
            num_outputs=num_outputs,
        )

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target,
            logits=q,
        )

        train_op = None
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=tf.contrib.framework.get_global_step(),
                decay_steps=100,
                decay_rate=0.96
            )

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=learning_rate,
                optimizer='Adam'
            )

        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions={
                'action_dist': q
            },
            loss=loss,
            train_op=train_op
        )

    def select(self, dataset, step: int):
        threshold = min(self.epsilon, step / 100.0)
        if threshold > random.random():
            prediction = self.estimator.predict(
                input_fn=lambda: self.input_fn(dataset)
            )

            prediction = list(prediction)[0]['action_dist']
            idx = np.argmax(prediction)

            return self.Action(idx)
        else:
            return random.choice(list(self.Action))

    def update(self, current, next, reward):
        if not os.path.exists('./model'):
            self.estimator.fit(
                input_fn=lambda: self.input_fn(current),
                steps=1
            )
        else:
            action = self.estimator.predict(
                input_fn=lambda: self.input_fn(current)
            )
            next_action = self.estimator.predict(
                input_fn=lambda: self.input_fn(next)
            )

            action = list(action)[0]['action_dist']
            next_action = list(next_action)[0]['action_dist']

            idx = np.argmax(next_action)
            action[idx] = reward + self.gamma * next_action[idx]

            self.estimator.fit(
                input_fn=lambda: self.input_fn(current, action),
                steps=1
            )


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

    commission = 0.0015
    budget = 10000
    num_coins = 0
    coin_value = 0

    for idx in range(len(files) - 1):
        state = files[idx]
        with open(state) as fp:
            orders = json.load(fp)['orders']

        portfolio = budget + num_coins * coin_value
        action = agent.select(orders, idx)
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
        next_state = files[idx + 1]
        with open(next_state) as fp:
            next_orders = json.load(fp)['orders']

        agent.update(orders, next_orders, reward)

        print('#%d (%s), Portfolio: %f, Budget: %d, Coin: %f(%d)' %
              (idx, action, (budget + num_coins * coin_value), budget, num_coins, coin_value))

    print('# Result')
    print('# Portfolio: %f' % (budget + num_coins * coin_value))
    print('# Budget: %d' % budget)
    print('# Number of coins: %f' % num_coins)
    print('# Value of a coin: %f' % coin_value)
