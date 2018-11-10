from common.secrets import SecretsManager
import requests
from text_classification_benchmarks.api_services.api_service import ApiService

BASE_URL = 'https://api.einstein.ai/v2/language/'
# noinspection SpellCheckingInspection
MODEL_ID = 'S3GGZAPKOLUB5GPBS4YF6QKTWU'
TRAINING_DATA_URL = 'https://s3-ap-southeast-2.amazonaws.com/txtclz/case_routing_intent.csv'


def create_dataset(train_df):
    train_df[['utterance', 'label']].to_csv('case_routing_intent.csv', header=False, index=False)


class EinsteinService(ApiService):

    def __init__(self, classes, max_api_calls=200, verbose=False):
        super().__init__(classes, max_api_calls, verbose)
        secrets = SecretsManager()
        jwt_token = secrets.get_secret('einstein/jwt_token')
        print(jwt_token)
        self.headers = {
            'Authorization': 'Bearer {}'.format(jwt_token),
            'Cache-Control': 'no-cache',
            'Content-Type': 'multipart/form-data'
        }
        self.dataset_id = None
        self.model_id = None

    def delete_model(self, model_id=None):
        model_id = model_id or self.model_id
        print('model_id:', model_id)
        response = requests.delete(BASE_URL + 'models/' + model_id, headers=self.headers)
        response_json = response.json()
        return response_json['status']

    def get_upload_status(self, dataset_id=None):
        dataset_id = dataset_id or self.dataset_id
        response = requests.get(BASE_URL + 'datasets/' + dataset_id, headers=self.headers)
        response_json = response.json()
        return response_json

    def get_training_status(self, model_id=None):
        model_id = model_id or self.model_id
        response = requests.get(BASE_URL + 'train/' + model_id, headers=self.headers)
        response_json = response.json()
        try:
            return response_json['status']
        except Exception as e:
            print('ERR:', e)
            print(response_json)
            return None

    def predict(self, utterance):
        payload = {'document': utterance, 'modelId': MODEL_ID}
        response = requests.post(BASE_URL + 'intent', data=payload, headers=self.headers)
        response_json = response.json()
        probabilities = []
        try:
            for hit in response_json['probabilities']:
                probabilities.append((hit['probability'], hit['label']))

        except Exception as e:
            print('ERR:', e)
            print(response_json)
            return None

        if len(probabilities) > 0:
            probabilities = sorted(probabilities, key=lambda x: -x[0])
            return probabilities[0][1]
        else:
            return None

    def train_model(self, dataset_id=None):
        dataset_id = dataset_id or self.dataset_id
        payload = {'name': 'Case Routing Model', 'datasetId': dataset_id}
        response = requests.post(BASE_URL + 'train', data=payload, headers=self.headers)
        response_json = response.json()
        self.model_id = response_json['modelId']
        return {
            'id': self.model_id,
            'name': response_json['name'],
            'status': response_json['status']
        }

    def upload_training_data(self):
        payload = {'path': TRAINING_DATA_URL, 'type': 'text-intent'}
        response = requests.post(BASE_URL + 'datasets/upload', data=payload, headers=self.headers)
        response_json = response.json()
        self.dataset_id = response_json['id']
        return {
            'id': self.dataset_id,
            'name': response_json['name'],
            'status': response_json['statusMsg']
        }
