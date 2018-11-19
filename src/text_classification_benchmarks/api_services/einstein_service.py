import jwt
import os
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from text_classification_benchmarks.api_services.api_service import ApiService
import time

BASE_URL = 'https://api.einstein.ai/v2/language/'


def create_dataset(train_df, filename='case_routing_intent.csv'):
    train_df[['utterance', 'label']].to_csv(filename, header=False, index=False)
    return os.path.abspath(filename)


class EinsteinService(ApiService):

    def __init__(self, classes, username=None, private_key_pem_filename=None,
                 model_id=None, max_api_calls=200, verbose=False):
        super().__init__(classes, max_api_calls, verbose)
        self.username = username
        self.private_key_pem_filename = private_key_pem_filename
        self.model_id = model_id
        self.expires = int(time.time())
        self.access_token = None
        self.dataset_id = None

    def delete_model(self, model_id=None):
        headers = {
            'Authorization': 'Bearer ' + self.get_access_token(),
            'Cache-Control': 'no-cache'
        }
        model_id = model_id or self.model_id
        response = requests.delete(BASE_URL + 'models/' + str(model_id), headers=headers)
        response_json = response.json()
        if response.ok:
            try:
                return response_json['status']
            except Exception as e:
                print('ERR:', e)
                return None

        else:
            print('ERR:', response_json['message'])
            return None

    def get_access_token(self):
        now = int(time.time())
        if now > self.expires - 10:  # Give 10 seconds to make the next call
            print('Fetching new access token')
            self.expires = now + 1800  # 30 min from now

            # 'aud' (audience) claim identifies the recipients that the JWT is intended for.
            # 'exp' (expiration time) claim identifies the expiration time on or after which
            # the JWT MUST NOT be accepted for processing.
            jwt_payload = {
                'sub': self.username,
                'aud': 'https://api.einstein.ai/v2/oauth2/token',
                'exp': self.expires
            }
            with open(self.private_key_pem_filename, 'r') as f:
                private_key = f.read()

            assertion = jwt.encode(jwt_payload, private_key, algorithm='RS256')
            payload = {
                'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                'assertion': assertion
            }
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            response = requests.post(jwt_payload['aud'], data=payload, headers=headers)
            response_json = response.json()
            self.access_token = response_json['access_token']

        return self.access_token

    def get_metrics(self, model_id=None):
        headers = {
            'Authorization': 'Bearer ' + self.get_access_token(),
            'Cache-Control': 'no-cache'
        }
        model_id = model_id or self.model_id
        response = requests.get(BASE_URL + 'models/' + str(model_id), headers=headers)
        response_json = response.json()
        if response.ok:
            try:
                return response_json['metricsData']
            except Exception as e:
                print('ERR:', e)
                return None

        else:
            print('ERR:', response_json['message'])
            return None

    def get_upload_status(self, dataset_id=None):
        headers = {
            'Authorization': 'Bearer ' + self.get_access_token(),
            'Cache-Control': 'no-cache'
        }
        dataset_id = dataset_id or self.dataset_id
        response = requests.get(BASE_URL + 'datasets/' + str(dataset_id), headers=headers)
        response_json = response.json()
        if response.ok:
            return response_json
        else:
            print('ERR:', response_json['message'])
            return None

    def get_training_status(self, model_id=None):
        headers = {
            'Authorization': 'Bearer ' + self.get_access_token(),
            'Cache-Control': 'no-cache'
        }
        model_id = model_id or self.model_id
        response = requests.get(BASE_URL + 'train/' + str(model_id), headers=headers)
        response_json = response.json()
        if response.ok:
            try:
                return response_json['status']
            except Exception as e:
                print('ERR:', e)
                return None

        else:
            print('ERR:', response_json['message'])
            return None

    def predict(self, utterance):
        multipart_data = MultipartEncoder(fields={'document': utterance, 'modelId': str(self.model_id)})
        headers = {
            'Authorization': 'Bearer ' + self.get_access_token(),
            'Content-Type': multipart_data.content_type
        }
        response = requests.post(BASE_URL + 'intent', data=multipart_data, headers=headers)
        response_json = response.json()
        if response.ok:
            probabilities = []
            try:
                for hit in response_json['probabilities']:
                    probabilities.append((hit['probability'], hit['label']))

            except Exception as e:
                print('ERR:', e)
                return None

        else:
            print('ERR:', response_json['message'])
            return None

        if len(probabilities) > 0:
            probabilities = sorted(probabilities, key=lambda x: -x[0])
            label = int(probabilities[0][1])
            return self.classes[label]
        else:
            print('ERR: no class probabilities')
            return None

    def train_model(self, dataset_id=None):
        dataset_id = dataset_id or self.dataset_id
        multipart_data = MultipartEncoder(fields={'name': 'Case Routing Model', 'datasetId': str(dataset_id)})
        headers = {
            'Authorization': 'Bearer ' + self.get_access_token(),
            'Cache-Control': 'no-cache',
            'Content-Type': multipart_data.content_type
        }
        response = requests.post(BASE_URL + 'train', data=multipart_data, headers=headers)
        response_json = response.json()
        if response.ok:
            try:
                self.model_id = response_json['modelId']
                return {
                    'id': self.model_id,
                    'name': response_json['name'],
                    'status': response_json['status']
                }
            except Exception as e:
                print('ERR:', e)
                return None

        else:
            print('ERR:', response_json['message'])
            return None

    def upload_training_data(self, training_data_url):
        if training_data_url.startswith(('http://', 'https://')):
            multipart_data = MultipartEncoder(fields={'path': training_data_url, 'type': 'text-intent'})
        else:
            multipart_data = MultipartEncoder(fields={'data': training_data_url, 'type': 'text-intent'})

        headers = {
            'Authorization': 'Bearer ' + self.get_access_token(),
            'Cache-Control': 'no-cache',
            'Content-Type': multipart_data.content_type
        }
        response = requests.post(BASE_URL + 'datasets/upload', data=multipart_data, headers=headers)
        response_json = response.json()
        if response.ok:
            try:
                self.dataset_id = response_json['id']
                return {
                    'id': self.dataset_id,
                    'name': response_json['name'],
                    'status': response_json['statusMsg']
                }
            except Exception as e:
                print('ERR:', e)
                return None

        else:
            print('ERR:', response_json['message'])
            return None
