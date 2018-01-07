import os


class Paths:
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    MODEL = os.path.join(ROOT, 'model')
    DATA = os.path.join(ROOT, 'data')
    CONFIG = os.path.join(ROOT, 'config.ini')
