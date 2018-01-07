from flask import Flask

from trading_service import TradingService

app = Flask(__name__)
service = TradingService(
    currency='xrp',
    budget=10000
)


@app.route("/resume")
def resume():
    service.resume()
    return "do_start"


@app.route("/pause")
def pause():
    service.pause()
    return "stop"


if __name__ == '__main__':
    service.start(mins=1, debug=True)
    app.run('localhost', 3000, debug=False)
