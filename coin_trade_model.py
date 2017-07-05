import configparser
import decimal
import gzip
import json
import os

import boto3
from boto3.dynamodb.conditions import Attr


class CoinTrade:
    class _DecimalEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            return super(CoinTrade._DecimalEncoder, self).default(obj)

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./aws.ini')

        self.__region = config['aws']['region']
        self.__access_key_id = config['aws']['access_key_id']
        self.__secret_access_key = config['aws']['secret_access_key']

    def download(self, currency: str):
        db = boto3.resource(service_name='dynamodb',
                            region_name=self.__region,
                            aws_access_key_id=self.__access_key_id,
                            aws_secret_access_key=self.__secret_access_key)
        trades_table = db.Table('trades')

        response = trades_table.scan(
            FilterExpression=Attr('currency').eq(currency)
        )

        if not os.path.exists('./out'):
            os.mkdir('./out')

        total = 0
        print('# %s' % currency)
        while True:
            items = response['Items']
            print('%d exists' % len(items))
            total += len(items)

            for item in items:
                with open(os.path.join('./out', item['key'] + '.json'), mode='w') as fp:
                    item['orders'] = json.loads(gzip.decompress(item['orders'].value).decode())
                    fp.write(json.dumps(item, cls=self._DecimalEncoder))

            if 'LastEvaluatedKey' in response:
                response = trades_table.scan(
                    FilterExpression=Attr('currency').eq(currency),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            else:
                break
        print('Total (%d)' % total)


if __name__ == '__main__':
    CoinTrade().download('eth')
