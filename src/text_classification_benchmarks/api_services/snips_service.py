import io
import json
import os
from snips_nlu import SnipsNLUEngine, load_resources
from snips_nlu.default_configs import CONFIG_EN
from text_classification_benchmarks.api_services.api_service import ApiService


def create_import_file(train_df, classes, output_path='.'):
    grouped = train_df.groupby(['label'])
    dataset_path = '{}/codi_dataset.json'.format(output_path)
    with open(dataset_path, 'w') as f:
        intents = {}
        for label, indices in grouped.groups.items():
            intent = classes[label]
            utterances = []
            for utterance in train_df.utterance.loc[indices].values:
                utterances.append({
                    'data': [
                        {
                            'text': utterance
                        }
                    ]
                })

            intents[intent] = {'utterances': utterances}

        data = {
            'entities': {},
            'intents': intents,
            'language': 'en'
        }
        json.dump(data, f)
        return data, os.path.abspath(dataset_path)


class SnipsService(ApiService):

    def __init__(self, classes, model_path=None, max_api_calls=None, verbose=False):
        super().__init__(classes, max_api_calls, verbose)
        load_resources('en')
        if model_path:
            self.load_model(model_path)
        else:
            self.engine = SnipsNLUEngine(config=CONFIG_EN)

    def train_model(self, dataset):
        self.engine.fit(dataset)

    def train_model_from_file(self, dataset_path):
        with io.open(dataset_path) as f:
            self.train_model(json.load(f))

    def save_model(self, model_path):
        self.engine.persist(model_path)

    def load_model(self, model_path):
        self.engine = SnipsNLUEngine.from_path(model_path)

    def predict(self, utterance):
        result = self.engine.parse(utterance)
        try:
            return result['intent']['intentName']
        except Exception as e:
            print('ERR:', e)
            print('Failed to parse: "{}"'.format(utterance))
            print(result)
            return None
