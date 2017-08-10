import argparse
import json
import os
from enum import Enum

import numpy as np
import tensorflow as tf


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# tf.logging.set_verbosity(tf.logging.INFO)


class DQN:
    def __init__(self, name: str, sess: tf.Session, params: dict):
        self.__name = name
        self.__sess = sess
        self.__build_network(name, params)

    def __build_network(self, name, params):
        with tf.variable_scope(name):
            self.__inputs = tf.placeholder(tf.float32, [None, params['input_size'], params['feature_size']],
                                           name='inputs')
            self.__targets = tf.placeholder(tf.float32, [None, params['output_size']], name='targets')

            outputs = tf.contrib.layers.fully_connected(
                inputs=self.__inputs,
                num_outputs=params['hidden_units'],
            )

            self.__q = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=params['output_size'],
            )

            self.__loss = tf.reduce_sum(tf.square(self.__targets - self.__q))
            self.__train = tf.contrib.layers.optimize_loss(
                loss=self.__loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=params['learning_rate'],
                optimizer='Adam'
            )

    def name(self):
        return self.__name

    def predict(self, state):
        return self.__sess.run(self.__q, feed_dict={
            self.__inputs: state
        })

    def update(self, states, targets):
        return self.__sess.run([self.__loss, self.__train], feed_dict={
            self.__inputs: states,
            self.__targets: targets
        })


class CoinAgent:

    class Action(Enum):
        NOTHING = 0
        BUY = 1
        SELL = 2

    actions = list(Action)
    num_outputs = len(actions)

    def __init__(self, params):
        self.commission = params['commission']
        self.budget = params['budget']
        self.num_coins = params['num_coins']
        self.coin_value = params['coin_value']

        self.e = params['e']
        self.r = params['r']
        self.max_length = params['max_length']
        self.hidden_units = params['hidden_units']
        self.learning_rate = params['learning_rate']

    def input_fn(self, dataset):
        name = dataset['name']
        states = dataset['states']
        targets = dataset['targets'] if 'targets' in dataset \
            else [[0.0 for _ in range(self.num_outputs)] for _ in range(len(states))]

        for i in range(len(states)):
            if self.max_length < len(states[i]):
                states[i] = states[i][len(states[i]) - self.max_length:]

        data = []
        for state in states:
            price = float(state['price'])
            qty = float(state['qty'])
            data.append([price, qty])

        if self.max_length > len(data):
            for _ in range(self.max_length - len(data)):
                data.insert(0, [0.0, 0.0])

        lengths = [len(state) for state in states]
        masks = [[1.0 if idx < length else 0.0 for idx in self.max_length] for length in lengths]

        inputs = {
            'name': name,
            'states': tf.constant(data),
            'lengths': tf.constant(lengths),
            'masks': tf.constant(masks)
        }

        targets = tf.constant(targets, dtype=tf.float32)

        return inputs, targets

    def __get_portfolio(self):
        return self.budget + self.num_coins * self.coin_value

    def __step(self, state, action):
        portfolio = self.__get_portfolio()

        self.coin_value = int(state[-1]['price'])
        if action == CoinAgent.Action.BUY and self.budget > 0:
            bought = (self.budget / self.coin_value) * (1 - self.commission)
            self.budget -= int(bought * self.coin_value)
            self.num_coins += bought
        elif action == CoinAgent.Action.SELL and self.num_coins > 0:
            self.budget += int(self.num_coins * self.coin_value * (1 - self.commission))
            self.num_coins = 0

        new_portfolio = self.__get_portfolio()
        return new_portfolio - portfolio

    @staticmethod
    def __get_copy_op(src: DQN, dest: DQN):
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src.name())
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest.name())

        ops = []
        for src_var, dest_var in zip(src_vars, dest_vars):
            ops.append(dest_var.assign(src_var.value()))

        return ops

    def train(self, trade_files):
        states = []
        targets = []

        params = {
            'input_size': self.max_length,
            'output_size': self.num_outputs,
            'hidden_units': self.hidden_units,
            'learning_rate': self.learning_rate
        }

        with tf.Session() as sess:
            mainDqn = DQN('main', sess, params)
            targetDqn = DQN('target', sess, params)
            sess.run(tf.global_variables_initializer())
            sess.run(self.__get_copy_op(mainDqn, targetDqn))

            for idx in range(len(trade_files)):
                with open(trade_files[idx]) as fp:
                    state = json.load(fp)['orders']

                action_dist = mainDqn.predict(state)
                action_max_idx = np.argmax(action_dist)
                action = CoinAgent.Action(action_max_idx)
                reward = self.__step(state, action)

                if idx == len(trade_files) - 1:
                    action_dist[action_max_idx] = reward
                else:
                    with open(trade_files[idx + 1]) as fp:
                        next_state = json.load(fp)['orders']
                    action_dist[action_max_idx] = reward + self.r * np.max(targetDqn.predict(next_state))

                print('#%d (%s), Portfolio: %f, Budget: %d, Coin: %f(%d)' %
                      (idx, action, self.__get_portfolio(), self.budget, self.num_coins, self.coin_value))

                states.append(state)
                targets.append(action_dist)

            mainDqn.update(states, targets)
            sess.run(self.__get_copy_op(mainDqn, targetDqn))

        return {
            'portfolio': self.__get_portfolio(),
            'budget': self.budget,
            'num_coins': self.num_coins,
            'coin_value': self.coin_value
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
        'commission': 0.0015,
        'budget': 10000,
        'num_coins': 0,
        'coin_value': 0,
        'e': 0.9,
        'r': 0.9,
        'hidden_units': 50,
        'learning_rate': 0.01,
        'gpu_memory': 0.1,
        'max_length': 12000
    })

    if not os.path.exists('./out'):
        os.mkdir('./out')

    trial = 10
    with open('./out/portfolio.txt', mode='w') as fp:
        for idx in range(trial):
            result = agent.train(files)
            msg = '#%d Portfolio: %f, Budget: %d, Coin: %f(%d)' % \
                  (idx, result['portfolio'], result['budget'], result['num_coins'], result['coin_value'])
            fp.write(msg + '\'n')
            print(msg)
