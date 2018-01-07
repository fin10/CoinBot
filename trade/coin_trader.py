import base64
import configparser
import hashlib
import hmac
import json
import time

import requests

from paths import Paths

config = configparser.ConfigParser()
config.read(Paths.CONFIG)


class CoinTrader:
    class URL:
        TICKER = 'https://api.coinone.co.kr/ticker/'
        TRADES = 'https://api.coinone.co.kr/trades/'
        BUY = 'https://api.coinone.co.kr/v2/order/limit_buy/'
        SELL = 'https://api.coinone.co.kr/v2/order/limit_sell/'

    __ACCESS_TOKEN = config['coinone']['access_token']
    __SECRET_KEY = config['coinone']['secret_key']

    @classmethod
    def __get_response(cls, url, payload):
        dumped_json = json.dumps(payload).encode(encoding='utf-8')
        encoded_payload = base64.b64encode(dumped_json)
        signature = hmac.new(cls.__SECRET_KEY.upper().encode(encoding='utf-8'), encoded_payload, hashlib.sha512)

        headers = {
            'Content-type': 'application/json',
            'X-COINONE-PAYLOAD': encoded_payload,
            'X-COINONE-SIGNATURE': signature.hexdigest()
        }

        response = requests.post(url, data=encoded_payload, headers=headers)
        return response

    @classmethod
    def __get_payload(cls, price, currency, qty):
        return {
            'access_token': cls.__ACCESS_TOKEN,
            'price': price,
            'qty': qty,
            'currency': currency,
            'nonce': int(time.time() * 1000)
        }

    @classmethod
    def get_trades(cls, currency):
        params = {
            'currency': currency,
            'period': 'hour'
        }

        response = requests.get(cls.URL.TRADES, params=params)
        if response.status_code != 200:
            raise ConnectionError(response.text)

        return json.loads(response.text)['completeOrders']

    @classmethod
    def ticker(cls, currency):
        oarams = {
            'currency': currency,
        }

        response = requests.get(cls.URL.TICKER, params=oarams)
        if response.status_code != 200:
            raise ConnectionError(response.text)

        return json.loads(response.text)

    @classmethod
    def limit_buy(cls, price, currency, qty, debug=False):
        if debug:
            return 200, 0

        payload = cls.__get_payload(price, currency, qty)
        res = cls.__get_response(cls.URL.BUY, payload)
        return res.status_code, int(json.loads(res.text)['errorCode'])

    @classmethod
    def limit_sell(cls, price, currency, qty, debug=False):
        if debug:
            return 200, 0

        payload = cls.__get_payload(price, currency, qty)
        res = cls.__get_response(cls.URL.SELL, payload)
        return res.status_code, int(json.loads(res.text)['errorCode'])
