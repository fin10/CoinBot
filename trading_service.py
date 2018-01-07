from apscheduler.schedulers.background import BackgroundScheduler

from coin_agent import CoinAgent
from logger import logger
from noti.slack_notification import SlackNotification
from trade.coin_trader import CoinTrader


class TradingService:

    def __init__(self, currency, budget):
        self.__commission = 0.0015
        self.__currency = currency
        self.__budget = budget
        self.__num_coin = 0.0
        self.__agent = CoinAgent(self.__commission)
        self.__scheduler = BackgroundScheduler()

    def start(self, mins=60, debug=False):
        self.__scheduler.add_job(lambda: self.do_trading(debug=debug), 'interval', minutes=mins, id='do_trading')
        self.__scheduler.start()

    def pause(self):
        self.__scheduler.pause()

    def resume(self):
        self.__scheduler.resume()

    def stop(self):
        self.__scheduler.shutdown()

    def do_trading(self, debug=False):
        logger.debug('debug=%s', debug)
        trades = CoinTrader.get_trades(self.__currency)
        prediction, dist = self.__agent.predict(trades, debug=debug)
        ticker = CoinTrader.ticker(self.__currency)
        latest_price = int(ticker['last'])
        qty = 0.0
        color = 'good'
        msg = 'Success'
        if prediction == CoinAgent.Action.BUY:
            if self.__budget > 10:
                qty = self.__budget / latest_price
                status_code, error_code = CoinTrader.limit_buy(latest_price, self.__currency, qty, debug=debug)
                if status_code == 200 and error_code == 0:
                    self.__budget -= latest_price * qty
                    self.__num_coin += qty * (1.0 - self.__commission)
                else:
                    color = 'danger'
                    msg = 'error: status_code: {}, error_code: {}'.format(status_code, error_code)
            else:
                color = 'warning'
                msg = 'Not enough budget'

        elif prediction == CoinAgent.Action.SELL:
            if self.__num_coin > 0.0001:
                qty = self.__num_coin
                status_code, error_code = CoinTrader.limit_sell(latest_price, self.__currency, qty, debug=debug)
                if status_code == 200 and error_code == 0:
                    self.__budget += latest_price * qty * (1.0 - self.__commission)
                    self.__num_coin -= qty
                else:
                    color = 'danger'
                    msg = 'error: status_code: {}, error_code: {}'.format(status_code, error_code)
            else:
                color = 'warning'
                msg = 'Not enough coins'

        logger.debug(
            '[{}] {} {} latest price {:,}, qty {:,}'.format(self.__currency, prediction, dist, latest_price, qty))

        SlackNotification.notify(
            title=self.__currency,
            color=color,
            msg='[{action}]\n'
                'Latest price {price:,} qty {qty:.4}\n'
                'Budget {budget}\n'
                'Coin {num_coin:.4}\n'
                '*{msg}*'.format(action=prediction,
                                 price=latest_price,
                                 qty=qty,
                                 budget=self.__budget,
                                 num_coin=self.__num_coin,
                                 msg=msg)
        )
