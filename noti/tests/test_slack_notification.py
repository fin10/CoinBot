import unittest

from noti.slack_notification import SlackNotification


class SlackNotificationTest(unittest.TestCase):

    def test_notify(self):
        res = SlackNotification.notify('test', 'good', 'test slack notification')
        self.assertEqual(200, res.status_code)
