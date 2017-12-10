import os
import random

import tensorflow as tf


# tf.logging.set_verbosity(tf.logging.INFO)


class DQN:

    def __init__(self, model_dir, output_size):
        self.__params = {
            'output_size': output_size,
            'batch_size': 1000,
            'max_length': 5000,
            'cell_size': 100,
            'learning_rate': 0.0001,
        }

        self.__model_path = model_dir
        self.__estimator = None

        if os.path.exists(self.__model_path):
            self.__estimator = self.__create_estimator()

    def __create_estimator(self):
        return tf.estimator.Estimator(
            model_fn=self.__model_fn,
            model_dir=self.__model_path,
            config=tf.estimator.RunConfig(
                save_summary_steps=10,
                save_checkpoints_steps=10,
            ),
            params=self.__params,
        )

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
            prev_price = -1
            prev_timestamp = -1

            for record in transaction:
                price = float(record['price'])
                qty = float(record['qty'])
                timestamp = float(record['timestamp'])

                if prev_price < 0 and prev_timestamp < 0:
                    prev_price = price
                    prev_timestamp = timestamp

                prices.append(price - prev_price)
                timestamps.append(timestamp - prev_timestamp)
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

    def __input_fn(self, transactions: list, action_dists: list = None, shuffle=False):
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
        cell_size = params['cell_size']
        output_size = params['output_size']
        learning_rate = params['learning_rate']
        keep_prob = 1.0 if mode != tf.contrib.learn.ModeKeys.TRAIN else 0.5

        prices = features['prices']
        qties = features['qties']
        timestamps = features['timestamps']
        length = features['length']

        inputs = tf.stack([prices, qties, timestamps], axis=2)
        inputs = tf.reshape(inputs, [-1, self.__params['max_length'] * 3])

        def dense(inputs, units):
            return tf.layers.dense(
                inputs=inputs,
                units=units,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        outputs = dense(inputs, cell_size)

        for _ in range(3):
            outputs = dense(outputs, cell_size)

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
        if not os.path.exists(self.__model_path):
            os.makedirs(self.__model_path)

        self.__estimator = self.__create_estimator()

        self.__estimator.train(
            input_fn=lambda: self.__input_fn(transactions, action_dists, shuffle=True)
        )

        return self.__estimator.evaluate(
            input_fn=lambda: self.__input_fn(transactions, action_dists)
        )

    def predict(self, transactions: list):
        if os.path.exists(self.__model_path):
            self.__estimator = self.__create_estimator()

        if self.__estimator is None:
            return [[random.random() for _ in range(self.__params['output_size'])] for _ in
                    range(len(transactions))]
        else:
            return list(self.__estimator.predict(
                input_fn=lambda: self.__input_fn(transactions)
            ))
