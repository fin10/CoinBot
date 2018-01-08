import configparser
import time

import requests

from paths import Paths

config = configparser.ConfigParser()
config.read(Paths.CONFIG)


class SlackNotification:
    __URL = config['slack']['url']

    @classmethod
    def notify(cls, title, color, msg, fields=None, actions=None):
        if actions is None:
            actions = []
        if fields is None:
            fields = []

        payload = {
            'attachments': [
                {
                    'title': title,
                    'color': color,
                    'fields': fields,
                    'actions': actions,
                    'text': msg,
                    'mrkdwn_in': ['text'],
                    'ts': time.time(),
                }
            ]
        }

        return requests.post(cls.__URL, json=payload)
