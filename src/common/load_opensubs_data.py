from common.load_cornell_data import split_conversations
from datetime import datetime, timedelta
from gzip import GzipFile
import os
import re
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ElementTree


# noinspection SpellCheckingInspection
def read_opensubs_data(dirname, max_len=20, fast_preprocessing=True):
    """
    Load the opensubtitles dialog corpus.

    Based on code from https://github.com/AlJohri/OpenSubtitles
    by Al Johri <al.johri@gmail.com>

    :param dirname:
    :param max_len:
    :param fast_preprocessing:
    :return:
    """
    dataset = OpenSubsData(dirname)
    conversations = dataset.get_conversations()
    return split_conversations(conversations, max_len=max_len, fast_preprocessing=fast_preprocessing)


class OpenSubsData:

    def __init__(self, dirname):
        print('Loading OpenSubtitles conversations in %s' % dirname)
        self.conversations = []
        self.tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
        self.conversations = self._load_conversations(dirname)

    def _load_conversations(self, dirname):
        conversations = []
        for filepath in tqdm(dir_files(dirname), 'OpenSubtitles data files'):
            if filepath.endswith('gz'):
                try:
                    doc = get_xml(filepath)
                    conversations.extend(self._gen_list(doc))
                except ValueError:
                    tqdm.write('Skipping file %s with errors' % filepath)
                except Exception:
                    print('Unexpected error:', sys.exc_info()[0])
                    raise

        return conversations

    def get_conversations(self):
        return self.conversations

    def _gen_list(self, tree):
        root = tree.getroot()
        time_format = '%H:%M:%S'
        max_delta = timedelta(seconds=1)
        start_time = datetime.min
        strbuf = ''
        sent_list = []
        for child in root:
            for el in child:
                if el.tag == 'time':
                    el_id = el.attrib['id']
                    el_val = el.attrib['value'][:-4]
                    if el_id[-1] == 'S':
                        start_time = datetime.strptime(el_val, time_format)
                    else:
                        sent_list.append((strbuf.strip(), start_time, datetime.strptime(el_val, time_format)))
                        strbuf = ''
                else:
                    # noinspection PyBroadException
                    try:
                        strbuf = strbuf + ' ' + el.text
                    except Exception:
                        pass

        conversations = []
        for i in range(0, len(sent_list) - 1):
            cur = sent_list[i]
            nxt = sent_list[i + 1]
            if cur and nxt and nxt[1] - cur[2] <= max_delta:
                tmp = {'lines': []}
                tmp['lines'].append(self.get_line(cur[0]))
                tmp['lines'].append(self.get_line(nxt[0]))
                if _filter(tmp):
                    conversations.append(tmp)

        return conversations

    def get_line(self, sentence):
        return {'text': self.tag_re.sub('', sentence).replace('\\\'', '\'').strip().lower()}


# noinspection PyUnusedLocal
def _filter(lines):
    """
    Use the following to customize filtering of QA pairs

    :param lines:
    :return:
    """
    # start_words = ['what', 'how', 'when', 'why', 'where', 'do', 'did',
    #                'is', 'are', 'can', 'could', 'would', 'will']
    # q = lines['lines'][0]['text']
    # if not q.endswith('?'):
    #     return False
    # if not q.split(' ')[0] in start_words:
    #     return False

    return True


def get_xml(filename):
    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        tmp = GzipFile(filename)
        return ElementTree.parse(tmp)
    else:
        return ElementTree.parse(filename)


def dir_files(dirname):
    result = []
    for dirpath, dirs, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(dirpath, filename)
            result.append(filepath)

    return result
