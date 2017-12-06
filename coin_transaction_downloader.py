import configparser
import decimal
import gzip
import json
import os
import time

import boto3
import botocore.exceptions
from boto3.dynamodb.conditions import Attr

from paths import Paths


class CoinTransactionDownloader:

    class _DecimalEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            return super(CoinTransactionDownloader._DecimalEncoder, self).default(obj)

    @classmethod
    def download(cls, currency: str):
        config = configparser.ConfigParser()
        config.read(os.path.join(Paths.ROOT, 'aws.ini'))

        region = config['aws']['region']
        access_key_id = config['aws']['access_key_id']
        secret_access_key = config['aws']['secret_access_key']

        db = boto3.resource(service_name='dynamodb',
                            region_name=region,
                            aws_access_key_id=access_key_id,
                            aws_secret_access_key=secret_access_key)
        trades_table = db.Table('trades')

        response = trades_table.scan(
            FilterExpression=Attr('currency').eq(currency)
        )

        if not os.path.exists(Paths.DATA):
            os.mkdir(Paths.DATA)

        total = 0
        print('# %s' % currency)
        while True:
            items = response['Items']
            total += len(items)

            for item in items:
                with open(os.path.join(Paths.DATA, item['key'] + '.json'), mode='w', encoding='utf-8') as fp:
                    item['orders'] = json.loads(gzip.decompress(item['orders'].value).decode())
                    fp.write(json.dumps(item, cls=cls._DecimalEncoder))

            if 'LastEvaluatedKey' in response:
                while True:
                    try:
                        response = trades_table.scan(
                            FilterExpression=Attr('currency').eq(currency),
                            ExclusiveStartKey=response['LastEvaluatedKey']
                        )
                        break
                    except botocore.exceptions.ClientError:
                        time.sleep(5)
            else:
                break

            print('%d downloaded.' % total)


if __name__ == '__main__':
    CoinTransactionDownloader.download('eth')
