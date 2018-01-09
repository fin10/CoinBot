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

    host = config['server']['host']
    port = config['server'].getint('port')

    currency = config['server']['currency']
    budget = config['server'].getint('budget')
    interval = config['server'].getint('interval')
    criteria = config['server'].getfloat('criteria')
    print('Currency: {}, Budget: {:,f}, Interval: {} mins, Criteria: {}%'.format(currency, budget, interval, criteria))

    service.start(currency, budget=budget, mins=interval, criteria=criteria)
    app.run(host, port, debug=False)
