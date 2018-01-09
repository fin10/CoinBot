import logging


class Logger:
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG)

    @staticmethod
    def get_logger():
        return logging.getLogger('coin.bot')


logger = Logger.get_logger()
