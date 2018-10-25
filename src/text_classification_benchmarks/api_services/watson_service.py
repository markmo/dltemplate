from common.secrets import SecretsManager
import pandas as pd
from text_classification_benchmarks.api_services.api_service import ApiService
from watson_developer_cloud import AssistantV1, WatsonApiException


def create_import_file(train_df, classes, output_path):
    import_df = pd.DataFrame({
        'utterance': train_df.utterance.values,
        'label': train_df.label.apply(lambda x: classes[x]).values},
        columns=['utterance', 'label'])
    import_df.to_csv(output_path, header=False, index=False)


class WatsonService(ApiService):

    def __init__(self, classes, max_api_calls=200, verbose=False):
        super().__init__(classes, max_api_calls, verbose)
        secrets = SecretsManager()
        self.workspace_id = secrets.get_secret('watson/workspace_id')
        self.assistant = AssistantV1(
            version='2018-09-20',
            iam_apikey=secrets.get_secret('watson/iam_apikey'),
            url='https://gateway-syd.watsonplatform.net/assistant/api'
        )

    def predict(self, utterance):
        try:
            # noinspection PyTypeChecker
            response = self.assistant.message(workspace_id=self.workspace_id, input={'text': utterance}).get_result()
            intents = sorted(response['intents'], key=lambda x: x['confidence'], reverse=True)
            return intents[0].get('intent') if intents else None

        except WatsonApiException as e:
            print('Error on utterance:', utterance)
            print('Method failed with status code {}: {}'.format(str(e.code), e.message))
        except Exception as e:
            print('Error on utterance:', utterance)
            print('Method failed: {}'.format(e))

        return None
