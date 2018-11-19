import boto3
import json
from num2words import num2words
import os
import re
from text_classification_benchmarks.api_services.api_service import ApiService
import time
import zipfile


def create_import_file(train_df, classes, output_path, bot_name):
    os.makedirs(output_path, exist_ok=True)
    intents_json = []
    grouped = train_df.groupby(['label'])
    all_utterances = {}
    too_long_count = 0
    duplicate_count = 0
    for label, indices in grouped.groups.items():
        intent = safe_list_get(classes[:100], label)
        if intent:
            utterances_json = []
            for utterance in train_df.utterance.loc[indices].values:
                utter = re.sub(r'(\s+|/+|_+)', ' ', utterance.strip())
                utter = re.sub(r'([0-9]+)', convert_numbers_to_words, utter)
                utter = re.sub(r'[^a-zA-Z.\- ]', '', utter)
                utter = re.sub(r'([.\-]){2,}', r'\1', utter)
                utter = re.sub(r'\s([.\-])', r'\1', utter)
                utter = re.sub(r'((^|\s)[.\-]+(\s|$))', '', utter)
                utter = utter.strip()[:200]
                if utter in all_utterances:
                    print('Utterance duplicated across intents, ignoring.')
                    duplicate_count += 1
                elif 1 < len(utter) <= 200:
                    utterances_json.append(utter)
                    all_utterances[utter] = True
                else:
                    print('Utterance too long, ignoring.')
                    too_long_count += 1

            intent_json = {
                'description': intent,
                'rejectionStatement': {
                    'messages': [
                        {
                            'contentType': 'PlainText',
                            'content': 'bye'
                        }
                    ]
                },
                'name': re.sub(r'[0-9]', 'D', intent.replace('-', '_')),
                'version': '1',
                'fulfillmentActivity': {
                    'type': 'ReturnIntent'
                },
                'sampleUtterances': list(set(utterances_json)),
                'slots': [],
                'confirmationPrompt': {
                    'messages': [
                        {
                            'contentType': 'PlainText',
                            'content': 'OK'
                        }
                    ],
                    'maxAttempts': 2
                }
            }
            intents_json.append(intent_json)

    del all_utterances
    print('Total duplicates:', duplicate_count, 'too long:', too_long_count)
    with open('{}/{}_Export.json'.format(output_path, bot_name), 'w') as f:
        import_json = {
            'metadata': {
                'schemaVersion': '1.0',
                'importType': 'LEX',
                'importFormat': 'JSON'
            },
            'resource': {
                'name': bot_name,
                'version': '1',
                'intents': intents_json,
                'voiceId': '0',
                'childDirected': False,
                'locale': 'en-US',
                'idleSessionTTLInSeconds': 300,
                'description': 'Benchmark Short Text Classification',
                'clarificationPrompt': {
                    'messages': [
                        {
                            'contentType': 'PlainText',
                            'content': 'Sorry, what can I help you with?'
                        }
                    ],
                    'maxAttempts': 2
                },
                'abortStatement': {
                    'messages': [
                        {
                            'contentType': 'PlainText',
                            'content': "Sorry, I'm not able to assist at this time"
                        }
                    ]
                }
            }
        }
        json.dump(import_json, f)

    zipf = zipfile.ZipFile('{}/{}_Bot_LEX_V1.zip'.format(output_path, bot_name), 'w', zipfile.ZIP_DEFLATED)
    zipf.write('{}/{}_Export.json'.format(output_path, bot_name))
    zipf.close()


class LexService(ApiService):

    def __init__(self, bot_name, bot_alias, classes, max_api_calls=200, verbose=False):
        super().__init__(classes, max_api_calls, verbose)
        self.bot_name = bot_name
        self.bot_alias = bot_alias
        self.short_classes = list(map(lambda x: x.replace('-', '_'), classes[:100].tolist()))
        self.client = boto3.client('lex-runtime', region_name='us-east-1')

    def predict(self, utterance):
        response = self.client.post_text(
            botName=self.bot_name,
            botAlias=self.bot_alias,
            userId='1234',
            inputText=utterance
        )
        return response['intentName']

    def predict_label(self, utterance):
        tic = time.time()
        intent = self.predict(utterance)
        toc = time.time()
        self.elapsed.append(toc - tic)
        return self.short_classes.index(intent) if intent else -1

    def predict_batch(self, val_df):
        y_pred = []
        j = 0
        for i, utterance in enumerate(val_df.utterance.values):
            y_true = self.classes[val_df.label.values[i]]
            y_true = y_true.replace('-', '_')
            if y_true in self.short_classes:
                j += 1
                label = self.predict_label(utterance)
                y_pred.append(label)
                if self.verbose:
                    print('Utterance: {}, Pred: {}, True: {}'.format(utterance, self.classes[label], y_true))
                    print()

                if self.max_api_calls and j > self.max_api_calls - 1:  # save on API calls
                    break

        return y_pred


def contains_alpha(text):
    return re.search(r'[a-zA-Z]', text)


def convert_numbers_to_words(match):
    group = match.group(1)
    return num2words(group)


def safe_list_get(l, idx, default=None):
    try:
        return l[idx]
    except IndexError:
        return default
