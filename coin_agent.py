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

    actions = list(Action)

    class DQN:
        def __init__(self, name: str, sess: tf.Session, params: dict):
            self.__name = name
            self.__sess = sess
            self.__feature_size = 2
            self.__params = params
            self.__build_network(name)

        def __build_network(self, name):
            input_size = self.__params['max_length']
            output_size = len(CoinAgent.actions)
            hidden_units = self.__params['hidden_units']
            learning_rate = self.__params['learning_rate']

            with tf.variable_scope(name):
                self.__inputs = tf.placeholder(tf.float32, [None, input_size, self.__feature_size], name='inputs')
                self.__lengths = tf.placeholder(tf.float32, [None], name='lengths')
                self.__masks = tf.placeholder(tf.float32, [None, input_size], name='masks')
                self.__targets = tf.placeholder(tf.float32, [None, output_size], name='targets')
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                outputs = tf.contrib.layers.fully_connected(
                    inputs=self.__inputs,
                    num_outputs=hidden_units,
                    scope='h1',
                )

                outputs = tf.reshape(outputs, [-1, input_size * hidden_units])

                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=output_size,
                    scope='h2',
                )

                self.__q = tf.nn.softmax(logits=outputs)

                self.__loss = tf.reduce_sum(tf.square(self.__targets - self.__q))
                self.__train = tf.contrib.layers.optimize_loss(
                    loss=self.__loss,
                    global_step=self.global_step,
                    learning_rate=learning_rate,
                    optimizer='Adam'
                )

        def __build_inputs(self, states):
            input_size = self.__params['max_length']

            for i in range(len(states)):
                if input_size < len(states[i]):
                    states[i] = states[i][len(states[i]) - input_size:]

            inputs = []
            for state in states:
                data = []
                for transaction in state:
                    price = float(transaction['price'])
                    qty = float(transaction['qty'])
                    data.append([price, qty])

                if input_size > len(data):
                    for _ in range(input_size - len(data)):
                        data.append([0.0, 0.0])
                inputs.append(data)

            lengths = [len(state) for state in states]
            masks = [[1.0 if idx < length else 0.0 for idx in range(input_size)] for length in lengths]

            return {
                'inputs': inputs,
                'lengths': lengths,
                'masks': masks
            }

        def name(self):
            return self.__name

        def predict(self, state):
            inputs = self.__build_inputs([state])

            return list(self.__sess.run(self.__q, feed_dict={
                self.__inputs: inputs['inputs'],
                self.__lengths: inputs['lengths'],
                self.__masks: inputs['masks']
            }))[0]

        def update(self, states, targets):
            inputs = self.__build_inputs(states)
            loss, _ = self.__sess.run([self.__loss, self.__train], feed_dict={
                self.__inputs: inputs['inputs'],
                self.__lengths: inputs['lengths'],
                self.__masks: inputs['masks'],
                self.__targets: targets
            })

            return loss

    def __init__(self, params):
        self.commission = params['commission']
        self.budget = params['budget']
        self.num_coins = params['num_coins']
        self.coin_value = params['coin_value']

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
        diff = new_portfolio - portfolio
        if diff > 0:
            return 1.0
        elif diff < 0:
            return -1.0
        else:
            return 0.0

    @staticmethod
    def __get_copy_op(src: DQN, dest: DQN):
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src.name())
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest.name())

        ops = []
        for src_var, dest_var in zip(src_vars, dest_vars):
            ops.append(dest_var.assign(src_var.value()))

        return ops

    @classmethod
    def train(cls, trade_files, params):
        e = params['e']
        r = params['r']

        with tf.Session() as sess:
            mainDqn = cls.DQN('main', sess, params)
            targetDqn = cls.DQN('target', sess, params)
            saver = tf.train.Saver(tf.global_variables())

            if not os.path.exists('./model'):
                os.mkdir('./model')

            ckpt = tf.train.get_checkpoint_state('./model')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reading variables from %s' % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Initializing variables..')
                sess.run(tf.global_variables_initializer())

            trial = 10
            for step in range(trial):
                agent = cls(params)
                sess.run(cls.__get_copy_op(mainDqn, targetDqn))

                states = []
                targets = []
                for idx in range(len(trade_files)):
                    with open(trade_files[idx]) as fp:
                        state = json.load(fp)['orders']

                    action_dist = mainDqn.predict(state)
                    action_max_idx = np.argmax(action_dist)

                    if random.random() < e:
                        action = CoinAgent.Action(action_max_idx)
                    else:
                        action = random.choice(cls.actions)

                    reward = agent.__step(state, action)

                    if idx == len(trade_files) - 1:
                        action_dist[action_max_idx] = reward
                    else:
                        with open(trade_files[idx + 1]) as fp:
                            next_state = json.load(fp)['orders']
                        action_dist[action_max_idx] = reward + r * np.max(targetDqn.predict(next_state))

                    if idx % 100 == 0:
                        print('[%d] (%s), Portfolio: %f, Budget: %d, Coin: %f(%d)' %
                              (idx, action, agent.__get_portfolio(), agent.budget, agent.num_coins, agent.coin_value))

                    states.append(state)
                    targets.append(action_dist)

                loss = mainDqn.update(states, targets)
                saver.save(sess, os.path.join('./model', 'coin_agent.ckpt'), global_step=mainDqn.global_step)

                result = {
                    'step': tf.train.global_step(sess, mainDqn.global_step),
                    'loss': loss,
                    'portfolio': agent.__get_portfolio(),
                    'budget': agent.budget,
                    'num_coins': agent.num_coins,
                    'coin_value': agent.coin_value
                }

                if not os.path.exists('./out'):
                    os.mkdir('./out')

                with open('./out/portfolio.txt', mode='a') as fp:
                    msg = '#%d Loss: %f, Portfolio: %f, Budget: %d, Coin: %f(%d)' % \
                                  (result['step'], result['loss'], result['portfolio'], result['budget'], result['num_coins'], result['coin_value'])
                    fp.write(msg + '\n')
                    print(msg)


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

    CoinAgent.train(files, params={
        'commission': 0.0015,
        'budget': 10000,
        'num_coins': 0,
        'coin_value': 0,
        'e': 0.9,
        'r': 0.9,
        'gpu_memory': 0.1,
        'max_length': 12000,
        'hidden_units': 50,
        'learning_rate': 0.01,
    })
