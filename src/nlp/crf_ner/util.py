from common.nlp_eval_util import nlp_metrics
import nltk


def build_sentence(tokens, tags):
    pos_tags = [it[-1] for it in nltk.pos_tag(tokens)]
    return list(zip(tokens, pos_tags, tags))


def build_sentences(tokens_set, tags_set):
    return [build_sentence(tokens, tags) for tokens, tags, in zip(tokens_set, tags_set)]


def evaluate(model, tokens, tags, short_report=True):
    tags_pred = model.predict(tokens)
    y_true = [y for s in tags for y in s]
    y_pred = [y for s in tags_pred for y in s]
    return nlp_metrics(y_true, y_pred, print_results=True, short_report=short_report)


def sentence_features(sent):
    return [token_features(sent, i) for i in range(len(sent))]


def sentence_labels(sent):
    return [label for _, _, label in sent]


def sentence_tokens(sent):
    return [token for token, _, _ in sent]


def token_features(sent, i):
    token, tag, _ = sent[i]
    features = {
        'bias': 1.0,
        'token_lower': token.lower(),
        'token_last_3_chars': token[-3:],
        'token_last_2_chars': token[-2:],
        'token_is_all_caps': token.isupper(),
        'token_is_title_case': token.istitle(),
        'token_is_digit': token.isdigit(),
        'tag': tag,
        'tag_prefix': tag[:2]
    }
    if i > 0:
        prev_token, prev_tag, _ = sent[i - 1]
        features.update({
            'prev_token_lower': prev_token.lower(),
            'prev_token_is_all_caps': prev_token.isupper(),
            'prev_token_is_title_case': prev_token.istitle(),
            'prev_tag': prev_tag,
            'prev_tag_prefix': prev_tag[:2]
        })
    else:
        features['start'] = True

    if i < len(sent) - 1:
        next_token, next_tag, _ = sent[i + 1]
        features.update({
            'next_token_lower': next_token.lower(),
            'next_token_is_all_caps': next_token.isupper(),
            'next_token_is_title_case': next_token.istitle(),
            'next_tag': next_tag,
            'next_tag_prefix': next_tag[:2]
        })
    else:
        features['end'] = True

    return features
