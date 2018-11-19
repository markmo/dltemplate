from common.secrets import SecretsManager
import http.client
import json
import numpy as np
import requests
from text_classification_benchmarks.api_services.api_service import ApiService
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = 'https://westus.api.cognitive.microsoft.com/luis/v2.0'
VERSION_ID = '0.1'


def create_import_file(train_df, classes, output_path=None, app_name='intent_test'):
    intents = []
    utterances_json = []
    grouped = train_df.groupby(['label'])
    for label, indices in grouped.groups.items():
        intent = classes[label]
        intents.append({'name': intent[:50]})
        for utterance in train_df.utterance.loc[indices].values:
            utterance_json = {
                'text': utterance[:50],  # LUIS has a text limit of 50 chars
                'intent': intent[:50],
                'entities': []
            }
            utterances_json.append(utterance_json)

    secrets = SecretsManager()
    sub_key = secrets.get_secret('luis/Ocp-Apim-Subscription-Key')
    # app_id = secrets.get_secret('luis/app_id')
    headers = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key': sub_key}
    # response = requests.post(BASE_URL + '/apps/{}/versions/{}/examples'.format(app_id, VERSION_ID),
    #                          data=json.dumps(utterances_json), headers=headers)
    # query_params = {'appName': 'intent_test'}
    body = {
        'luis_schema_version': '3.0.0',
        'versionId': '0.1',
        'name': app_name,
        'desc': 'Benchmark Short Text Classification',
        'culture': 'en-us',
        'intents': intents,
        'entities': [],
        'composites': [],
        'closedLists': [],
        'patternAnyEntities': [],
        'regex_entities': [],
        'prebuiltEntities': [],
        'model_features': [],
        'regex_features': [],
        'patterns': [],
        'utterances': utterances_json
    }
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(body, f)

    # response = requests.post(BASE_URL + '/apps/import', params=query_params, data=body, headers=headers)
    # pprint(vars(response))
    query_params = urllib.parse.urlencode({'appName': app_name})
    try:
        conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
        conn.request('POST', '/luis/api/v2.0/apps/import?%s' % query_params, json.dumps(body), headers)
        response = conn.getresponse()
        data = response.read()
        print(data)
        conn.close()
    except Exception as e:
        print('Error:', e)


class LuisService(ApiService):

    def __init__(self, classes, max_api_calls=200, verbose=False, app_id=None):
        if type(classes) is np.ndarray:
            classes = classes.tolist()

        classes = list(map(lambda x: x[:50], classes))
        super().__init__(classes, max_api_calls, verbose)
        secrets = SecretsManager()
        sub_key = secrets.get_secret('luis/Ocp-Apim-Subscription-Key')
        app_id = app_id or secrets.get_secret('luis/app_id')
        self.url = '{}/apps/{}?subscription-key={}&timezoneOffset=-360&q=%s'.format(BASE_URL, app_id, sub_key)

    def predict(self, utterance):
        response = requests.get(self.url % utterance)
        response_json = response.json()
        return response_json.get('topScoringIntent', {}).get('intent')
