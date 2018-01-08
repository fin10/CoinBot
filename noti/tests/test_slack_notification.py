import unittest

from coin_agent import CoinAgent
from noti.slack_notification import SlackNotification


class SlackNotificationTest(unittest.TestCase):

    def test_notify(self):
        res = SlackNotification.notify(
            title='xrp',
            color='good',
            msg='[{action}] {dist}\n'
                '가격 {price:,} 거래량 {qty:,.4f}\n'
                '*{msg}*'.format(action=CoinAgent.Action.BUY,
                                 dist=[0, 0, 0],
                                 price=10000,
                                 qty=1000.0999,
                                 msg='Success'),
            fields=[
                {
                    'title': '지갑',
                    'value': 'KRW {budget:,}\n'
                             'Coin {coin:,.4f}\n'
                             '자산 {portfolio:,}'.format(budget=10000, coin=10.95, portfolio=100000),
                    'short': True
                }, {
                    'title': '수익률',
                    'value': '일일 {daily:,f}% 기준 {daily_budget:,}\n'
                             '전체 {total:,f}% 기준 {total_budget:,}'.format(daily=15.0, total=20.0,
                                                                         daily_budget=1000, total_budget=12000),
                    'short': True
                }]
        )
        self.assertEqual(200, res.status_code)
