import configparser
import time

import requests

from paths import Paths

config = configparser.ConfigParser()
config.read(Paths.CONFIG)


class SlackNotification:
    __URL = config['slack']['url']

    @classmethod
    def notify(cls, title, color, msg):
        payload = {
            'attachments': [
                {
                    'title': title,
                    'color': color,
                    'text': msg,
                    'mrkdwn_in': ['text'],
                    'ts': time.time()
                }
            ]
        }
        return requests.post(cls.__URL, json=payload)
