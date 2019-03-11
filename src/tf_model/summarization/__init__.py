from argparse import ArgumentParser
from common.load_data import DATA_DIR
from common.model_util import load_hyperparams, merge_dict
import pandas as pd
from pathlib import Path
import pickle
from tf_model.summarization.model_setup import build_models
from tf_model.summarization.util import clean_text, prepare_data, train
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


def run(constant_overwrites):
    config_path = ROOT_DIR / 'hyperparams.yml'
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    dataset_path = ROOT_DIR / 'data' / 'reviews_dataset.pkl'
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            stories = pickle.load(f)
    else:
        stories = load_data()

    stories = stories[:constants['n_samples']]
    (encoder_input_data, decoder_input_data, decoder_target_data,
     n_encoder_tokens, n_decoder_tokens) = prepare_data(stories)
    model, encoder_model, decoder_model = build_models(n_encoder_tokens, n_decoder_tokens, constants['n_hidden'])
    model_dir = ROOT_DIR / 'model'
    train(model, encoder_input_data, decoder_input_data, decoder_target_data,
          constants['n_epochs'], constants['batch_size'], model_dir)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Summarization Model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int, help='dimension of RNN hidden states')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--samples', dest='n_samples', type=int, help='number training cases to sample')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--train', dest='train', help='training mode', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
