import ast
import nltk
import os
import re
from tqdm import tqdm


def read_cornell_data(dirname, max_len=20, fast_preprocessing=True):
    """
    Load the Cornell movie dialog corpus.

    Available from:
    http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

    :param dirname:
    :param max_len:
    :param fast_preprocessing:
    :return:
    """
    dataset = CornellData(dirname)
    conversations = dataset.get_conversations()
    return split_conversations(conversations, max_len=max_len, fast_preprocessing=fast_preprocessing)


def split_conversations(conversations, max_len=20, fast_preprocessing=True):
    data = []
    for conv in tqdm(conversations):
        lines = conv['lines']
        for i in range(len(lines) - 1):
            request = extract_text(lines[i]['text'], fast_preprocessing=fast_preprocessing)
            reply = extract_text(lines[i + 1]['text'], fast_preprocessing=fast_preprocessing)
            if 0 < len(request) <= max_len and 0 < len(reply) <= max_len:
                data += [(request, reply)]

    return data


def extract_text(line, fast_preprocessing=True):
    if fast_preprocessing:
        good_symbols_re = re.compile('[^0-9a-z ]')
        replace_by_space_re = re.compile('[/(){}\[\]|@,;#+_]')
        replace_several_spaces = re.compile('\s+')
        line = line.lower()
        line = replace_by_space_re.sub(' ', line)
        line = good_symbols_re.sub('', line)
        line = replace_several_spaces.sub(' ', line)
        return line.strip()
    else:
        return nltk.word_tokenize(line)


def load_lines(filename, fields):
    """

    :param filename: file to load
    :param fields: (set<str>) fields to extract
    :return: dict<dict<str>> the extracted fields for each line
    """
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')

            # Extract fields
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]

            lines[line_obj['lineID']] = line_obj

    return lines


class CornellData(object):

    def __init__(self, dirname):
        """

        :param dirname: directory of corpus
        """
        movie_lines_fields = ['lineID', 'characterID', 'movieID', 'character', 'text']
        movie_conversations_fields = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']
        self.lines = load_lines(os.path.join(dirname, 'movie_lines.txt'), movie_lines_fields)
        self.conversations = self._load_conversations(os.path.join(dirname, 'movie_conversations.txt'),
                                                      movie_conversations_fields)

    def _load_conversations(self, filename, fields):
        conversations = []
        with open(filename, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(' +++$+++ ')

                # Extract fields
                conv_obj = {}
                for i, field in enumerate(fields):
                    conv_obj[field] = values[i]

                # Convert string to list (conv_obj['utteranceIDs'] == "['L598485', 'L598486', ...]")
                line_ids = ast.literal_eval(conv_obj['utteranceIDs'])

                # Reassemble lines
                conv_obj['lines'] = []
                for line_id in line_ids:
                    conv_obj['lines'].append(self.lines[line_id])

                conversations.append(conv_obj)

        return conversations

    def get_conversations(self):
        return self.conversations
