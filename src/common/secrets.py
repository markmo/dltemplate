import hvac
import os


class SecretsManager(object):

    def __init__(self):
        print(os.environ['VAULT_TOKEN'])
        self.client = hvac.Client(url='http://127.0.0.1:8200', token=os.environ['VAULT_TOKEN'])

    def get_secret(self, key):
        print('Get secret...')
        namespace, key = key.split('/')
        print('namespace: "{}", key: "{}"'.format(namespace, key))
        response = self.client.read('secret/' + namespace)
        return response.get('data', {}).get(key) if response else None
