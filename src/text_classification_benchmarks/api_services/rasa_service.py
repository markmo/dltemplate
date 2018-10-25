from rasa_nlu.model import Interpreter
from text_classification_benchmarks.api_services.api_service import ApiService


def create_import_file(train_df, classes, output_path='.'):
    grouped = train_df.groupby(['label'])
    with open('{}/nlu.md'.format(output_path), 'w') as f:
        for label, indices in grouped.groups.items():
            f.write('## intent:{}\n'.format(classes[label]))
            for utterance in train_df.utterance.loc[indices].values:
                f.write('- {}\n'.format(utterance))

            f.write('\n')


class RasaService(ApiService):

    def __init__(self, model_path, classes, max_api_calls=None, verbose=False):
        super().__init__(classes, max_api_calls, verbose)
        self.interpreter = Interpreter.load(model_path)

    def predict(self, utterance):
        result = self.interpreter.parse(utterance)
        return result['intent']['name']
