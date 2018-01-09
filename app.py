import configparser

from flask import Flask

from paths import Paths
from trading_service import TradingService

app = Flask(__name__)
service = TradingService()


@app.route("/")
def index():
    return 'Hello Coin Bot!'


@app.route("/resume")
def resume():
    try:
        service.resume()
    except Exception as e:
        return str(e)
    return "Server resumed"


@app.route("/pause")
def pause():
    try:
        service.pause()
    except Exception as e:
        return str(e)
    return "Server stopped"


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(Paths.CONFIG)

    currency = config['server']['currency']
    budget = config['server']['budget']
    interval = config['server']['interval']
    criteria = config['server']['criteria']
    print('Currency: {}, Budget: {:,}, Interval: {} mins, Criteria: {}%'.format(currency, budget, interval, criteria))

    service.start(currency, budget=budget, mins=interval, criteria=criteria)
    app.run(config['server']['host'], config['server']['port'], debug=False)
