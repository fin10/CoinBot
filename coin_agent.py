import argparse
import json
import os
import random
from collections import deque
from enum import Enum

import numpy as np
import tensorflow as tf


# tf.logging.set_verbosity(tf.logging.INFO)


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
                self.__lengths = tf.placeholder(tf.int32, [None], name='lengths')
                self.__masks = tf.placeholder(tf.float32, [None, input_size], name='masks')
                self.__targets = tf.placeholder(tf.float32, [None, output_size], name='targets')
                self.__global_step = tf.Variable(0, name='global_step', trainable=False)

                def make_cell(size):
                    cell = tf.contrib.rnn.GRUCell(size)
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
                    return cell

                cell_fw = tf.contrib.rnn.MultiRNNCell([make_cell(hidden_units) for _ in range(2)])
                cell_bw = tf.contrib.rnn.MultiRNNCell([make_cell(hidden_units) for _ in range(2)])

                activations, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=self.__inputs,
                    sequence_length=self.__lengths,
                    dtype=tf.float32
                )

                activations = activations[0] + activations[1]
                outputs = tf.reshape(activations, [-1, input_size * hidden_units])

                self.__q = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=output_size,
                )

                self.__loss = tf.reduce_sum(tf.square(self.__targets - self.__q))
                self.__train = tf.contrib.layers.optimize_loss(
                    loss=self.__loss,
                    global_step=self.__global_step,
                    learning_rate=learning_rate,
                    optimizer='Adam'
                )

                tf.summary.scalar('loss', self.__loss)
                tf.summary.scalar('step', self.__global_step)
                self.__merged = tf.summary.merge_all()

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

        def predict(self, states):
            inputs = self.__build_inputs(states)

            return list(self.__sess.run(self.__q, feed_dict={
                self.__inputs: inputs['inputs'],
                self.__lengths: inputs['lengths'],
                self.__masks: inputs['masks']
            }))

        def update(self, states, targets):
            inputs = self.__build_inputs(states)
            summary, loss, step, _ = self.__sess.run([self.__merged, self.__loss, self.__global_step, self.__train],
                                                     feed_dict={
                                                         self.__inputs: inputs['inputs'],
                                                         self.__lengths: inputs['lengths'],
                                                         self.__masks: inputs['masks'],
                                                         self.__targets: targets
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
            rewords.append(new_portfolio - portfolio)

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
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
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
            sample_size = 500
            for step in range(trial):
                agent = cls(params)
                queue = deque(maxlen=sample_size)
                sess.run(cls.__get_copy_op(main_dqn, target_dqn))

                for idx in range(len(states)):
                    queue.append(states[idx])

                    if (idx == len(states) - 1) or (idx > 0 and idx % sample_size == 0):
                        action_dists = main_dqn.predict(list(queue))
                        action_max_indices = np.argmax(action_dists, axis=1)
                        actions = [CoinAgent.Action(index) if random.random() > e else random.choice(cls.actions)
                                   for index in action_max_indices]

                        next_action_dists = target_dqn.predict(list(queue)[1:])
                        nex_action_maxes = np.max(next_action_dists, axis=1)

                        rewards = agent.__step(list(queue), actions)
                        for i in range(len(queue)):
                            action_max_index = action_max_indices[i]
                            if i == len(queue) - 1:
                                action_dists[i][action_max_index] = rewards[i]
                            else:
                                action_dists[i][action_max_index] = rewards[i] + r * nex_action_maxes[i]

                        print('Updating...')
                        sample_indices = random.choices([i for i in range(len(queue))], k=int(sample_size * 0.1))
                        samples = (
                            [queue[index] for index in sample_indices],
                            [action_dists[index] for index in sample_indices]
                        )

                        summary, loss, global_step = main_dqn.update(samples[0], samples[1])
                        writer.add_summary(summary, global_step=global_step)
                        print('[{}] Loss: {:,.2f}, Portfolio: {:,.2f}'.format(idx, loss, agent.__get_portfolio()))

                        saver.save(sess, os.path.join('./model', 'coin_agent.ckpt'), global_step=global_step)

                        result = {
                            'step': global_step,
                            'loss': loss,
                            'portfolio': agent.__get_portfolio(),
                            'budget': agent.budget,
                            'num_coins': agent.num_coins,
                            'coin_value': agent.coin_value
                        }

                if not os.path.exists('./out'):
                    os.mkdir('./out')

                with open('./out/portfolio.txt', mode='a') as fp:
                    msg = '#{:<4} Loss: {:>20,.2f}, Portfolio: {:>15,.2f}, Budget: {}, Coin: {}({})'.format(
                        result['step'], result['loss'],
                        result['portfolio'], result['budget'], result['num_coins'], result['coin_value'])
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
        'budget': 10000000,
        'num_coins': 0,
        'coin_value': 0,
        'e': 0.1,
        'r': 0.9,
        'max_length': 10000,
        'hidden_units': 50,
        'learning_rate': 0.1,
    })
