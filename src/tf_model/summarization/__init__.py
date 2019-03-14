from argparse import ArgumentParser
from common.load_data import DATA_DIR
from common.model_util import load_hyperparams, merge_dict
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tf_model.summarization.model_setup import build_models
from tf_model.summarization.util import clean_text, decode_sequence, prepare_data
from tf_model.summarization.util import train
import yaml

ROOT_DIR = Path(__file__).parent


def load_data():
    data = pd.read_csv(Path(DATA_DIR) / 'summarization' / 'amazon-fine-food-reviews' / 'Reviews.csv')
    data = data.dropna()
    data = data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
                      'HelpfulnessDenominator', 'Score', 'Time'], 1)
    data = data.reset_index(drop=True)
    print('Loaded data')
    with open(ROOT_DIR / 'contractions.yml') as f:
        config = yaml.load(f)

    contractions = config['contractions']
    print('Loaded contractions')
    clean_summaries = []
    for summary in data.Summary:
        clean_summaries.append(clean_text(summary, contractions, remove_stopwords=False))

    print('Prepared summaries')
    clean_texts = []
    for text in data.Text:
        clean_texts.append(clean_text(text, contractions, remove_stopwords=True))

    print('Prepared text')
    stories = []
    for i, text in enumerate(clean_texts):
        stories.append({'story': text, 'highlights': clean_summaries[i]})

    with open(ROOT_DIR / 'data' / 'reviews_dataset.pkl', 'wb') as f:
        pickle.dump(stories, f)

    return stories


def load_test_data():
    data = pd.read_csv(Path(DATA_DIR) / 'summarization' / 'amazon-fine-food-reviews' / 'Reviews_test.csv')
    data = data.dropna()
    data = data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
                      'HelpfulnessDenominator', 'Score', 'Time'], 1)
    data = data.reset_index(drop=True)
    print('Loaded test data')
    with open(ROOT_DIR / 'contractions.yml') as f:
        config = yaml.load(f)

    contractions = config['contractions']
    print('Loaded contractions')
    clean_summaries = []
    for summary in data.Summary:
        clean_summaries.append(clean_text(summary, contractions, remove_stopwords=False))

    print('Prepared summaries')
    clean_texts = []
    for text in data.Text:
        clean_texts.append(clean_text(text, contractions, remove_stopwords=True))

    print('Prepared text')
    stories = []
    for i, text in enumerate(clean_texts):
        stories.append({'story': text, 'highlights': clean_summaries[i]})

    with open(ROOT_DIR / 'data' / 'reviews_test_dataset.pkl', 'wb') as f:
        pickle.dump(stories, f)

    return stories


def run(constant_overwrites):
    config_path = ROOT_DIR / 'hyperparams.yml'
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    model_dir = ROOT_DIR / 'model'
    if constants['train']:
        print('Training...')
        dataset_path = ROOT_DIR / 'data' / 'reviews_dataset.pkl'
        if dataset_path.exists():
            with open(dataset_path, 'rb') as f:
                stories = pickle.load(f)
        else:
            stories = load_data()

        stories = stories[:constants['n_samples']]
        (encoder_input_data, decoder_input_data, decoder_target_data,
         n_encoder_tokens, n_decoder_tokens, target_token_index,
         reverse_input_char_index, reverse_target_char_index,
         max_dec_seq_length, input_texts) = prepare_data(stories)
        model, encoder_model, decoder_model = build_models(n_encoder_tokens, n_decoder_tokens, constants['n_hidden'])
        train(model, encoder_input_data, decoder_input_data, decoder_target_data,
              constants['n_epochs'], constants['batch_size'], model_dir)

        for seq_i in range(10):
            # Take one sequence (part of the training set) for trying out decoding
            input_seq = encoder_input_data[seq_i: seq_i + 1]
            decoded = decode_sequence(input_seq, encoder_model, decoder_model, n_decoder_tokens,
                                      target_token_index, reverse_target_char_index, max_dec_seq_length)
            print('-')
            print('Input sentence:', input_texts[seq_i])
            print('Decoded sentence:', decoded)
    else:
        print('Evaluating...')
        # dataset_path = ROOT_DIR / 'data' / 'reviews_test_dataset.pkl'
        # if dataset_path.exists():
        #     with open(dataset_path, 'rb') as f:
        #         stories = pickle.load(f)
        # else:
        #     stories = load_test_data()
        #
        # stories = stories[:constants['n_test_samples']]
        # with open(ROOT_DIR / 'data' / 'lengths.csv', 'r') as f:
        #     n_encoder_tokens, n_decoder_tokens, _, _ = f.read().rstrip('\n').split(',')
        #
        # n_encoder_tokens = int(n_encoder_tokens)
        # n_decoder_tokens = int(n_decoder_tokens)
        # model, encoder_model, decoder_model = build_models(n_encoder_tokens, n_decoder_tokens, constants['n_hidden'])
        # model.load_weights(str(model_dir / constants['model_name']))
        # total, correct = 100, 0
        # for _ in range(total):
        #     x1, x2, y = get_eval_dataset(n_encoder_tokens, n_decoder_tokens, n_features, 1)
        #     target = predict_sequence(x1, encoder_model, decoder_model, n_decoder_tokens, n_features)
        #     if np.array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
        #         correct += 1
        #
        # print('Accuracy: %.2f%%' % float(correct / total * 100))
        #
        # # Spot check some examples
        # for _ in range(10):
        #     x1, x2, y = get_eval_dataset(n_encoder_tokens, n_decoder_tokens, n_features, 1)
        #     target = predict_sequence(x1, encoder_model, decoder_model, n_decoder_tokens, n_features)
        #     print('X=%s y=%s, y_hat=%s' % (one_hot_decode(x1[0]), one_hot_decode(y[0]), one_hot_decode(target)))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Summarization Model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int, help='dimension of RNN hidden states')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--samples', dest='n_samples', type=int, help='number training cases to sample')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--model', dest='model_name', type=str, help='name of saved model to evaluate')
    parser.add_argument('--train', dest='train', help='training mode', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
