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

"""
Setup information:
------------------
1. Create an Azure resource and resource group. Sign up for an Azure account.
   * Create new resource: search for 'luis', select 'Language Understanding', click 'Create'
   * Enter Name, Subscription: 'Pay-As-You-Go', Location: (must be a region supported by
     LUIS, 'Australia East' no good, I selected 'US West'), Pricing Tier: 'F0 (5 Calls per second,
     10K Calls per month)', Select or create new Resource Group, then click 'Create'
2. Sign up for LUIS at https://www.luis.ai/ using the same Microsoft account and
   email address
3. Under 'My Apps', click 'Import new app'
4. After import, click 'Train'
5. Click 'Manage':
   * Note the 'Application ID' under 'Application Information'. This will be needed
     to call the service.
   * Click on 'Keys and Endpoints' in the left menu
   * At the bottom, click on 'Assign resource'
   * Enter Tenant name: your Azure account, Subscription Name: 'Pay-As-You-Go',
     LUIS resource name: the name of resource created above, then click 'Assign resource'
6. The resource is added to the table below. Copy one of the keys. This will be needed
   to call the service. Also note the URL endpoint: use this to call the service.
"""


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
        # time.sleep(.2)  # LUIS rate limit of 5TPS
        return response_json.get('topScoringIntent', {}).get('intent')
