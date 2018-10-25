from common.secrets import SecretsManager
from common.util import zipdir
import json
import os
import requests
from text_classification_benchmarks.api_services.api_service import ApiService
import time
import zipfile

BASE_URL = 'https://api.dialogflow.com/v1/'


def create_import_file(train_df, classes, output_path):
    os.makedirs('{}/intents'.format(output_path), exist_ok=True)
    grouped = train_df.groupby(['label'])
    for label, indices in grouped.groups.items():
        intent = classes[label]
        with open('{}/intents/{}.json'.format(output_path, intent), 'w') as f:
            intent_json = {
                # 'id': str(uuid.uuid4()),
                'name': intent,
                'auto': True,
                'contexts': [],
                'responses': [
                    {
                        'resetContexts': False,
                        'affectedContexts': [],
                        'parameters': [],
                        'messages': [
                            {
                                'type': 0,
                                'lang': 'en',
                                'speech': []
                            }
                        ],
                        'defaultResponsePlatforms': {},
                        'speech': []
                    }
                ],
                'priority': 500000,
                'webhookUsed': False,
                'webhookForSlotFilling': False,
                'lastUpdate': 1481334512,
                'fallbackIntent': False,
                'events': []
            }
            json.dump(intent_json, f)

        with open('{}/intents/{}_usersays_en.json'.format(output_path, intent), 'w') as f:
            utterances_json = []
            for utterance in train_df.utterance.loc[indices].values:
                utterance_json = {
                    # 'id': str(uuid.uuid4()),
                    'data': [
                        {
                            'text': utterance,
                            'userDefined': False
                        }
                    ],
                    'isTemplate': False,
                    'count': 0,
                    'updated': 1538295035
                }
                utterances_json.append(utterance_json)

            json.dump(utterances_json, f)

    zip_path = '{}/dialogflow_import.zip'.format(os.path.abspath(os.path.join(output_path, os.pardir)))
    zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    zipdir(output_path, zipf)
    zipf.close()


class DialogflowService(ApiService):

    def __init__(self, classes, max_api_calls=200, verbose=False):
        super().__init__(classes, max_api_calls, verbose)
        secrets = SecretsManager()
        client_access_token = secrets.get_secret('dialogflow/client_access_token')
        self.headers = {'Authorization': 'Bearer {}'.format(client_access_token)}

    def predict(self, utterance):
        payload = {'query': utterance, 'v': '20150910', 'lang': 'en', 'sessionId': 12345}
        response = requests.get(BASE_URL + 'query', params=payload, headers=self.headers)
        response_json = response.json()
        return response_json.get('result', {}).get('metadata', {}).get('intentName')

    def predict_label(self, utterance):
        tic = time.time()
        intent = self.predict(utterance)
        toc = time.time()
        self.elapsed.append(toc - tic)
        if intent and intent != 'Default Fallback Intent':
            return self.classes.index(intent)
        else:
            return -1
