import configparser
import time

from apscheduler.schedulers.background import BackgroundScheduler

from agent.coin_agent import CoinAgent
from logger import logger
from noti.slack_notification import SlackNotification
from paths import Paths
from trade.coin_trader import CoinTrader


class TradingService:
    __config = configparser.ConfigParser()
    __config.read(Paths.CONFIG)

    __commission = 0.0015

    def __init__(self):
        self.__currency = None
        self.__budget = 0
        self.__num_coin = 0.0
        self.__criteria = 0.0
        self.__agent = CoinAgent(self.__commission)
        self.__scheduler = BackgroundScheduler()

        self.__portfolio = None

    def start(self, currency, budget, criteria=-50.0, mins=60):
        self.__currency = currency
        self.__budget = budget
        self.__criteria = criteria

        self.__portfolio = {
            'principal': budget,
            'daily': budget,
            'time': time.localtime()
        }

        self.__scheduler.add_job(self.do_trading, 'interval', minutes=mins, id='do_trading')
        self.__scheduler.start()

        SlackNotification.notify('Coin Bot', 'good', 'Server started!', fields=[{
                'title': '지갑',
                'value': 'KRW {budget:,}\n'.format(budget=self.__budget),
                'short': True
        }])

        self.do_trading()

    def pause(self):
        self.__scheduler.pause()
        SlackNotification.notify('Coin Bot', 'good', 'Server paused!')

    def resume(self):
        self.__scheduler.resume()
        SlackNotification.notify('Coin Bot', 'good', 'Server resumed!')

    def stop(self):
        self.__scheduler.shutdown(False)
        SlackNotification.notify('Coin Bot', 'good', 'Server Stopped!')

    def result_notify(self, color, prediction, dist, latest_price, qty, msg, portfolio, daily, total):
        SlackNotification.notify(
            title=self.__currency,
            color=color,
            msg='[{action}] {dist}\n'
                '가격 {price:,} 거래량 {qty:,.4f}\n'
                '*{msg}*'.format(action=prediction,
                                 dist=dist,
                                 price=latest_price,
                                 qty=qty,
                                 msg=msg),
            fields=[
                {
                    'title': '지갑',
                    'value': 'KRW {budget:,}\n'
                             'Coin {coin:,.4f}\n'
                             '자산 {portfolio:,.4f}'.format(budget=self.__budget, coin=self.__num_coin,
                                                          portfolio=portfolio),
                    'short': True
                }, {
                    'title': '수익률',
                    'value': '일일 {daily:,.4f}% 기준 {daily_budget:,.4f}\n'
                             '전체 {total:,.4f}% 기준 {total_budget:,.4f}'.format(daily=daily, total=total,
                                                                              daily_budget=self.__portfolio['daily'],
                                                                              total_budget=self.__portfolio['principal']
                                                                              ),
                    'short': True
                }],
        )

    def do_trading(self):
        prediction_debug = self.__config['debug'].getboolean('prediction')
        payment_debug = self.__config['debug'].getboolean('payment')
        logger.info('Debug: prediction %s, payment %s', prediction_debug, payment_debug)

        trades = CoinTrader.get_trades(self.__currency)
        prediction, dist = self.__agent.predict(trades, debug=prediction_debug)
        ticker = CoinTrader.ticker(self.__currency)
        latest_price = int(ticker['last'])
        qty = 0.0
        color = 'good'
        msg = 'Success'
        if prediction == CoinAgent.Action.BUY:
            if self.__budget > 10:
                qty = self.__budget / latest_price
                status_code, error_code = CoinTrader.limit_buy(latest_price, self.__currency, qty, debug=payment_debug)
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
                status_code, error_code = CoinTrader.limit_sell(latest_price, self.__currency, qty, debug=payment_debug)
                if status_code == 200 and error_code == 0:
                    self.__budget += latest_price * qty * (1.0 - self.__commission)
                    self.__num_coin -= qty
                else:
                    color = 'danger'
                    msg = 'error: status_code: {}, error_code: {}'.format(status_code, error_code)
            else:
                color = 'warning'
                msg = 'Not enough coins'

        portfolio = self.__budget + self.__num_coin * latest_price

        cur_time = time.localtime()
        if cur_time.tm_mday != self.__portfolio['time'].tm_mday:
            self.__portfolio['daily'] = portfolio
            self.__portfolio['time'] = cur_time

        daily = 100.0 * ((portfolio / self.__portfolio['daily']) - 1.0)
        total = 100.0 * ((portfolio / self.__portfolio['principal']) - 1.0)

        logger.debug('[{}] {} {} portfolio {:,} latest price {:,}, qty {:,}'.format(
            self.__currency, prediction, dist, portfolio, latest_price, qty
        ))

        self.result_notify(color=color, prediction=prediction, dist=dist, latest_price=latest_price, qty=qty, msg=msg,
                           portfolio=portfolio, daily=daily, total=total)

        if total < self.__criteria:
            SlackNotification.notify(
                'Coin Bot', 'danger', '전체 수익률이 {}% 이하로 떨어져 강제 종료 합니다.'.format(self.__criteria))
            self.stop()
