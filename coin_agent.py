import argparse
import json
import os
import random
from collections import deque, Counter
from enum import Enum

import numpy as np
import tensorflow as tf


# tf.logging.set_verbosity(tf.logging.INFO)


class CoinAgent:
    class Action(Enum):
        BUY = 0
        SELL = 1
        NOTHING = 2

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
                self.__lengths = tf.placeholder(tf.int32, [None], name='lengths')
                self.__targets = tf.placeholder(tf.float32, [None, output_size], name='targets')
                self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                self.__global_step = tf.Variable(0, name='global_step', trainable=False)

                # def cell(size):
                #     cell = tf.contrib.rnn.GRUCell(size)
                #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.__keep_prob)
                #     return cell

                # cell_fw = cell(hidden_units)
                # cell_bw = cell(hidden_units)

                # activations, _ = tf.nn.bidirectional_dynamic_rnn(
                #     cell_fw=cell_fw,
                #     cell_bw=cell_bw,
                #     inputs=self.__inputs,
                #     sequence_length=self.__lengths,
                #     dtype=tf.float32
                # )
                #
                # activations = activations[0] + activations[1]

                outputs = tf.contrib.layers.fully_connected(
                    inputs=self.__inputs,
                    num_outputs=hidden_units,
                )

                for _ in range(3):
                    outputs = tf.contrib.layers.fully_connected(
                        inputs=outputs,
                        num_outputs=hidden_units,
                    )
                    outputs = tf.nn.dropout(outputs, keep_prob=self.__keep_prob)

                outputs = tf.reshape(outputs, [-1, input_size * hidden_units])

                self.__q = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=output_size,
                    activation_fn=None
                )

                self.__q = tf.nn.softmax(self.__q)

                learning_rate = tf.train.exponential_decay(
                    learning_rate=learning_rate,
                    global_step=self.__global_step,
                    decay_steps=100,
                    decay_rate=0.96
                )

                self.__loss = tf.losses.mean_squared_error(self.__targets, self.__q)
                self.__train = tf.contrib.layers.optimize_loss(
                    loss=self.__loss,
                    global_step=self.__global_step,
                    learning_rate=learning_rate,
                    optimizer='Adam'
                )

                tf.summary.scalar('learning_rate', learning_rate)
                self.__merged = tf.summary.merge_all()

        def __build_inputs(self, states):
            input_size = self.__params['max_length']

            for i in range(len(states)):
                if input_size < len(states[i]):
                    indices = random.choices([j for j in range(len(states[i]))], k=input_size)
                    indices.sort()
                    states[i] = [states[i][j] for j in indices]

            inputs = []
            for state in states:
                data = []
                base_price = -1
                for transaction in state:
                    price = float(transaction['price'])
                    qty = float(transaction['qty'])
                    if base_price < 0:
                        base_price = price
                        price = 0.0
                    else:
                        price = (price - base_price) / base_price

                    data.append([price, qty])

                if input_size > len(data):
                    for _ in range(input_size - len(data)):
                        data.append([0.0, 0.0])
                inputs.append(data)

            lengths = [len(state) for state in states]

            return {
                'inputs': inputs,
                'lengths': lengths
            }

        def name(self):
            return self.__name

        def predict(self, states, keep_prob=1.0):
            inputs = self.__build_inputs(states)

            return list(self.__sess.run(self.__q, feed_dict={
                self.__inputs: inputs['inputs'],
                self.__lengths: inputs['lengths'],
                self.__keep_prob: keep_prob
            }))

        def update(self, states, targets):
            inputs = self.__build_inputs(states)
            summary, loss, step, _ = self.__sess.run([self.__merged, self.__loss, self.__global_step, self.__train],
                                                     feed_dict={
                                                         self.__inputs: inputs['inputs'],
                                                         self.__lengths: inputs['lengths'],
                                                         self.__targets: targets,
                                                         self.__keep_prob: 0.5
                                                     })

            return summary, loss, step

    def __init__(self, params):
        self.commission = params['commission']
        self.budget = params['budget']
        self.num_coins = params['num_coins']
        self.coin_value = params['coin_value']

    def __get_portfolio(self):
        return self.budget + self.num_coins * self.coin_value

    def __step(self, states, actions):
        rewords = []
        for state, action in zip(states, actions):
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
            # rewords.append((new_portfolio - portfolio) / portfolio)
            diff = new_portfolio - portfolio
            if diff > 0:
                rewords.append(1.0)
            elif diff < 0:
                rewords.append(-1.0)
            else:
                rewords.append(0.0)

        return rewords

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
        states = []

        print('Loading dataset...')
        for idx in range(len(trade_files)):
            with open(trade_files[idx]) as fp:
                states.append(json.load(fp)['orders'])
        print('Completed.')

        if not os.path.exists('./model'):
            os.mkdir('./model')

        with tf.Session() as sess:
            main_dqn = cls.DQN('main', sess, params)
            target_dqn = cls.DQN('target', sess, params)
            copy_ops = cls.__get_copy_op(main_dqn, target_dqn)
            saver = tf.train.Saver(tf.global_variables(),
                                   # max_to_keep=50
                                   )
            writer = tf.summary.FileWriter('./model', sess.graph)

            ckpt = tf.train.get_checkpoint_state('./model')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reading variables from %s' % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Initializing variables..')
                sess.run(tf.global_variables_initializer())

            print('Training starts.')

            result = {}
            trial = 50
            sample_size = params['sample_size']
            batch_size = params['batch_size']
            for _ in range(trial):
                agent = cls(params)
                queue = deque(maxlen=sample_size)

                sess.run(copy_ops)
                print('Models copied.')

                for idx in range(len(states)):
                    queue.append(states[idx])

                    if (idx == len(states) - 1) or (idx > 0 and idx % sample_size == 0):
                        action_dists = main_dqn.predict(queue, keep_prob=0.5)
                        action_max_indices = np.argmax(action_dists, axis=1)
                        actions = [CoinAgent.Action(index) if random.random() > e else random.choice(cls.actions)
                                   for index in action_max_indices]

                        next_action_dists = target_dqn.predict(list(queue)[1:], keep_prob=0.5)
                        next_action_maxes = np.max(next_action_dists, axis=1)

                        rewards = agent.__step(queue, actions)
                        for i in range(len(queue)):
                            if i == len(queue) - 1:
                                action_dists[i][action_max_indices[i]] = rewards[i]
                            else:
                                action_dists[i][action_max_indices[i]] = rewards[i] + r * next_action_maxes[i]

                        sample_indices = random.choices([i for i in range(len(queue))], k=batch_size)
                        samples = (
                            [queue[index] for index in sample_indices],
                            [action_dists[index] for index in sample_indices]
                        )

                        summary, loss, global_step = main_dqn.update(samples[0], samples[1])
                        writer.add_summary(summary, global_step=global_step)
                        print('[{}] Loss: {:>20,.2f}, Portfolio: {:>20,.2f}'.format(idx, loss, agent.__get_portfolio()))

                        saver.save(sess, os.path.join('./model', 'coin_agent.ckpt'), global_step=global_step)

                        result = {
                            'step': global_step,
                            'loss': loss,
                        }

                if not os.path.exists('./out'):
                    os.mkdir('./out')

                with open('./out/portfolio.txt', mode='a') as fp:
                    test_agent = cls(params)
                    action_dists = []
                    for chunk in np.array_split(states, 2):
                        action_dists += main_dqn.predict(chunk)

                    action_max_indices = np.argmax(action_dists, axis=1)
                    actions = [CoinAgent.Action(index) for index in action_max_indices]
                    test_agent.__step(states, actions)

                    msg = '#{:<4} Loss: {:>20,.2f}, Portfolio: {:>20,.2f}, {}'.format(
                        result['step'], result['loss'], test_agent.__get_portfolio(), Counter(actions))
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
        'budget': 100000,
        'num_coins': 0,
        'coin_value': 0,
        'e': 0.1,
        'r': 0.99,
        'max_length': 5000,
        'hidden_units': 100,
        'learning_rate': 0.1,
        'sample_size': 1500,
        'batch_size': 100
    })
