import random

import matplotlib.pyplot as plt

from coin_agent import CoinAgent
from paths import Paths


class TransactionVisualizer:

    def __init__(self, currency):
        self.__transactions = CoinAgent.get_transactions(Paths.DATA, currency)

    @staticmethod
    def __get_values(transaction, count=-1):
        times = []
        prices = []
        qties = []

        if count > 0:
            indices = random.sample([n for n in range(len(transaction))], k=count)
            indices.sort()
            transaction = [transaction[i] for i in indices]

        first_time = -1
        first_price = -1
        for record in transaction:
            time = int(record['timestamp'])
            price = float(record['price'])
            qty = float(record['qty'])

            if first_price < 0 or first_time < 0:
                first_time = time
                first_price = price

            times.append((time - first_time) / 3600.0)
            prices.append(((price / first_price) - 1.0) * 50 + 1.0)
            qties.append(qty)

        return times, prices

    def show(self, idx=0):
        transaction = self.__transactions[idx]
        print('[%d] transaction size: %d' % (idx, len(transaction)))

        plt.subplot(221)
        plt.title('all')
        pair = self.__get_values(transaction)
        plt.plot(pair[0], pair[1])

        plt.subplot(222)
        plt.title('5000')
        pair = self.__get_values(transaction, 5000)
        plt.plot(pair[0], pair[1])

        plt.subplot(223)
        plt.title('1000')
        pair = self.__get_values(transaction, 1000)
        plt.plot(pair[0], pair[1])

        plt.subplot(224)
        plt.title('100')
        pair = self.__get_values(transaction, 100)
        plt.plot(pair[0], pair[1])

        plt.show()


if __name__ == '__main__':
    visualizer = TransactionVisualizer('eth')
    visualizer.show()
