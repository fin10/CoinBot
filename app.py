from flask import Flask

from trading_service import TradingService

app = Flask(__name__)
service = TradingService(
    currency='xrp',
    budget=10000,
    criteria=-50.0
)


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
    service.start(mins=60, debug=True)
    app.run('localhost', 3000, debug=False)
