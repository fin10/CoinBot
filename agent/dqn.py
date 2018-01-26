import os
import random

import tensorflow as tf


# tf.logging.set_verbosity(tf.logging.INFO)


class DQN:

    def __init__(self, model_dir, output_size):
        self.__params = {
            'output_size': output_size,
            'batch_size': 5000,
            'max_length': 100,
            'rnn_cell_size': 50,
            'nn_cell_size': 50,
            'learning_rate': 0.0001,
        }

        self.__model_path = model_dir
        self.__estimator = self.__create_estimator()

        if not os.path.exists(self.__model_path):
            os.makedirs(self.__model_path)
            self.__estimator.train(
                input_fn=lambda: self.__input_fn()
            )

    def __create_estimator(self):
        estimator = tf.estimator.Estimator(
            model_fn=self.__model_fn,
            model_dir=self.__model_path,
            config=tf.estimator.RunConfig(
                save_summary_steps=10,
                save_checkpoints_steps=10,
            ),
            params=self.__params,
        )

        return estimator

    def generate_input(self, transactions: list, action_dists: list):
        max_length = self.__params['max_length']

        for transaction, action_dist in zip(transactions, action_dists):
            if max_length < len(transaction):
                indices = random.sample([n for n in range(len(transaction))], k=max_length)
                indices.sort()
                transaction = [transaction[i] for i in indices]

            prices = []
            qties = []
            timestamps = []
            first_price = -1
            first_timestamp = -1

            for record in transaction:
                price = float(record['price'])
                qty = float(record['qty'])
                timestamp = float(record['timestamp'])

                if first_price < 0 and first_timestamp < 0:
                    first_price = price
                    first_timestamp = timestamp

                prices.append((((price / first_price) - 1.0) * 50 + 1.0))
                timestamps.append((timestamp - first_timestamp) / 3600.0)
                qties.append(qty)

            if len(prices) < max_length:
                diff = max_length - len(prices)
                prices += [0.0 for _ in range(diff)]
                qties += [0.0 for _ in range(diff)]
                timestamps += [0.0 for _ in range(diff)]

            yield {
                      'prices': prices,
                      'qties': qties,
                      'timestamps': timestamps,
                      'length': len(transaction)
                  }, action_dist

    def __input_fn(self, transactions: list = None, action_dists: list = None, shuffle=False):
        if transactions is None:
            transactions = [[]]

        batch_size = self.__params['batch_size'] if self.__params['batch_size'] > 0 else len(transactions)
        output_size = self.__params['output_size']
        max_length = self.__params['max_length']

        if action_dists is None:
            action_dists = [[0.0 for _ in range(output_size)] for _ in range(len(transactions))]

        dataset = tf.data.Dataset.from_generator(
            lambda: self.generate_input(transactions, action_dists),
            ({'prices': tf.float32, 'qties': tf.float32, 'timestamps': tf.float32, 'length': tf.int32}, tf.float32),
            ({
                 'prices': tf.TensorShape([max_length]),
                 'qties': tf.TensorShape([max_length]),
                 'timestamps': tf.TensorShape([max_length]),
                 'length': tf.TensorShape([])
             }, tf.TensorShape([output_size]))
        )

        if shuffle:
            dataset = dataset.shuffle(batch_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)

        iterator = dataset.make_one_shot_iterator()
        features, label = iterator.get_next()

        return features, label

    def __model_fn(self, features, labels, mode, params):
        rnn_cell_size = params['rnn_cell_size']
        nn_cell_size = params['nn_cell_size']
        output_size = params['output_size']
        learning_rate = params['learning_rate']
        keep_prob = 1.0 if mode != tf.contrib.learn.ModeKeys.TRAIN else 0.5

        prices = features['prices']
        qties = features['qties']
        timestamps = features['timestamps']
        length = features['length']

        inputs = tf.stack([prices, timestamps], axis=2)

        def rnn_cell(cell_size):
            cell = tf.nn.rnn_cell.GRUCell(
                num_units=cell_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell(rnn_cell_size),
            cell_bw=rnn_cell(rnn_cell_size),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32
        )

        outputs = outputs[0] + outputs[1]
        outputs = tf.reshape(outputs, [-1, self.__params['max_length'] * rnn_cell_size])

        def dense(inputs, units):
            return tf.layers.dense(
                inputs=inputs,
                units=units,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        for _ in range(3):
            outputs = dense(outputs, nn_cell_size)
        outputs = dense(outputs, output_size)

        loss = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = tf.losses.mean_squared_error(
                labels=labels,
                predictions=outputs
            )

        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=outputs,
            loss=loss,
            train_op=train_op,
        )

    def train(self, transactions: list, action_dists: list):
        self.__estimator.train(
            input_fn=lambda: self.__input_fn(transactions, action_dists, shuffle=True)
        )

        return self.__estimator.evaluate(
            input_fn=lambda: self.__input_fn(transactions, action_dists)
        )

    def predict(self, transactions: list):
        predictions = list(self.__estimator.predict(
            input_fn=lambda: self.__input_fn(transactions)
        ))
        return [list(p) for p in predictions]
