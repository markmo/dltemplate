import hvac
import os


class SecretsManager(object):

    def __init__(self):
        self.client = hvac.Client(url='http://localhost:8200', token=os.environ['VAULT_TOKEN'])

    def get_secret(self, key):
        namespace, key = key.split('/')
        response = self.client.read('secret/' + namespace)
        return response.get('data', {}).get(key) if response else None
