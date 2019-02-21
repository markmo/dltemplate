from abc import ABC, abstractmethod
import csv
from collections import defaultdict
from enum import Enum
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import re
import spacy
import string
import sys
import tensorflow as tf
import time
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union


class Vocabulary(object):

    class SpecialTokens(object):

        EMPTY = '<empty>'       # not a word

        UNKNOWN = '<unk>'       # unknown word, aka OOV (out-of-vocabulary)

        END_OF_STRING = '<eos>'

        all = [EMPTY, UNKNOWN, END_OF_STRING]

    def __init__(self, my_iter):
        """
        Builds a vocabulary using the words from `my_iter`,
        keeping only words that appear at least `min_count` times.

        :param my_iter: (generator) of words
        """
        counted = self._count_words(my_iter)
        self._forward, self._backward = self._assign_indexes(counted)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other._forward == self._forward and other._backward == self._backward

        return False

    @staticmethod
    def _count_words(my_iter) -> Dict[str, int]:
        """Loop through `my_iter` and count words"""
        count: Dict = {}
        for word in my_iter:
            count[word] = count.get(word, 0) + 1

        return count

    @staticmethod
    def _queue_names(queue_file: str = os.path.join(os.path.dirname(__file__), 'queues.json')) -> List[str]:
        """Loads a queue list for adding tokens to the vocab. Defaults to "queues.json" in the current dir"""
        return json.load(open(queue_file))

    @staticmethod
    def _generate_backward(forward: Dict[int, str]) -> Dict[str, int]:
        """Shallow flip of `forward` dict"""
        return {v: k for k, v in forward.items()}

    def _assign_indexes(self, counted_words: Dict[str, int]) -> (Dict[int, str], Dict[str, int]):
        """Assigns indexes to words with higher count => lower idx"""
        forward = {}

        def add_word(word: str):
            if word not in forward:
                forward[len(forward)] = word

        for w in self.SpecialTokens.all:
            add_word(w)

        for w in self._queue_names():
            add_word(w)

        sorted_words = [k for k, _ in sorted(counted_words.items(), key=lambda x: x[1], reverse=True)]
        for w in sorted_words:
            add_word(w)

        backward = self._generate_backward(forward)
        return forward, backward

    def tokens_and_indices(self) -> Tuple[List[str], List[int]]:
        """Returns all tokens and their indices in this vocabulary in ordered sequence"""
        sorted_items = sorted(self._forward.items(), key=lambda x: x[0])
        return [v[1] for v in sorted_items], [k[0] for k in sorted_items]

    def token2idx(self, token: str) -> int:
        """Get the vocab index for a token. Return index for Unknown token if not found."""
        return self._backward.get(token, self._backward[self.SpecialTokens.UNKNOWN])

    def transform(self, sentence: Iterable[str]) -> List[int]:
        """
        Transforms a `sentence` into a list of ints representing the indices
        of the tokens within this vocab.

        Unknown words are mapped to index 1 ('<unk>').

        Inverse of `lookup`.

        :param sentence: list of strings
        :return:
        """
        return [self.token2idx(w) for w in sentence]

    def idx2token(self, idx: int) -> str:
        """
        Get the token for an index. Return Unknown token if not found.

        :param idx:
        :return:
        """
        return self._forward.get(idx, self.SpecialTokens.UNKNOWN)

    def lookup(self, idxs: Iterable[int]) -> List[str]:
        """
        Transforms a list of ints into a sentence (list of strings).

        Unknown indexes are mapped to '<unk>'.

        Inverse of `transform`.

        :param idxs:
        :return:
        """
        return [self.idx2token(idx) for idx in idxs]

    def size(self) -> int:
        """Returns the number of words in the vocabulary"""
        return len(self._forward)

    def dump(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self._forward, f)

    def load(self, filename: str):
        d = json.load(open(filename, 'r')).items()

        # Need to convert from str back to int due to json format
        self._forward = {int(k): v for k, v in d}
        self._backward = self._generate_backward(self._forward)

    # Handy shortcuts
    @property
    def empty_token_index(self):
        return self.token2idx(self.SpecialTokens.EMPTY)

    @property
    def end_of_string_token_index(self):
        return self.token2idx(self.SpecialTokens.END_OF_STRING)

    @property
    def unknown_token_index(self):
        return self.token2idx(self.SpecialTokens.UNKNOWN)


def load_vocab_json(filename: str) -> Vocabulary:
    vocab = Vocabulary([])
    vocab.load(filename)
    return vocab


nlp = None
accepted_chars = set(string.printable) - {',', '&', '<'}
int_re = re.compile(r'[1-9]')
quote_mark_re = re.compile(r'"')
slash_re = re.compile(r'\\')
agent_name_re = re.compile(r'\${agentname}')
cust_name_re = re.compile(r'\${customername}')


def clean(raw: str, for_export: bool) -> str:
    """Removes unwanted and non-ascii characters, converts to lower case, and zeros digits"""
    text = raw.strip().lower()
    filtered = [x for x in text if x in accepted_chars]
    cleaned = int_re.sub('0', ''.join(filtered))
    cleaned = quote_mark_re.sub('"', cleaned)
    cleaned = slash_re.sub(r'\\', cleaned)
    if for_export:
        cleaned = agent_name_re.sub('##agentname##', cleaned)
        cleaned = cust_name_re.sub('##customername##', cleaned)

    return cleaned


def replace_x_0(raw: str) -> str:
    """Replace 'xx/xx' with '00/00'"""
    out = []
    for word in raw.split():
        alpha = re.findall(r'[A-Za-z]', word)
        if all(a == 'x' for a in alpha):
            w = re.sub(r'x', '0', word)
            out.append(w)
        else:
            out.append(word)

    return ' '.join(out)


def process_response(raw: str) -> str:
    """
    TODO - combine with `clean` method

    * convert to lower case
    * remove double quotes
    * replace digits with '0'
    * remove commas
    * replace '(customer name)' with 'xnamex'
    * replace 'xx/xx' with '00/00'

    :param raw: (str) text to be cleaned
    :return:
    """
    text = raw.strip().lower()
    cleaned = re.sub(r'"', '', text)
    cleaned = re.sub(r'\d', '0', cleaned)
    cleaned = re.sub(r',', '', cleaned)
    cleaned = re.sub(r'\(customer name\)', 'xnamex', cleaned)
    cleaned = replace_x_0(cleaned)
    return cleaned


def to_word_list(raw: str, for_export: bool = False) -> List[str]:
    """Returns cleaned, tokenized list of words from `raw`. Uses spacy library."""
    return tokenize(clean(raw, for_export))


def tokenize(cleaned: str) -> List[str]:
    global nlp
    if nlp is None:
        nlp = spacy.load('en', parser=False)
    tokenized = [token.text.strip() for token in nlp(cleaned)]
    tokenized = [token for token in tokenized if token]
    return tokenized


class StringEncoder(object):

    def __init__(self, vocab: Union[str, Vocabulary], max_len=0, for_export=False):
        if isinstance(vocab, Vocabulary):
            self._vocab = vocab
        else:
            self._vocab = load_vocab_json(vocab)

        self.max_len = max_len
        self.for_export = for_export

    def add_eos_token(self, tokens: List[str]) -> List[str]:
        return tokens + [self._vocab.SpecialTokens.END_OF_STRING]

    def transform_to_vocab_idx_add_eos(self, word_list: List[str]) -> List[int]:
        return self._vocab.transform(self.add_eos_token(word_list))

    def fixed_len(self, a_string: str) -> Tuple[List[int], int]:
        """
        Creates a fixed length integer representation of a string.

        Pads with '<eos>' token and then '<empty>' tokens until `max_len`.

        :param a_string: (str) arbitrary string
        :return:
            list: of integers
            length: length of string
        """
        assert self.max_len > 0, 'Not initialized with `max_len`'
        word_list = to_word_list(a_string, for_export=self.for_export)
        padding = [self._vocab.empty_token_index] * (self.max_len - len(word_list) - 1)
        padded = self.transform_to_vocab_idx_add_eos(word_list) + padding

        return padded[0:self.max_len], len(word_list) + 1

    def fixed_len_list(self, strings: List[str]):
        """
        As `fixed_len` but expects a list of strings, and runs `fixed_len`
        on each element.

        :param strings: (list[str]) list of arbitrary strings
        :return:
            lists: (list[list[int]]) encoded
            lengths: parallel list[int] length of original string
        """
        lists, lengths = zip(*[self.fixed_len(x) for x in strings])
        return lists, lengths

    @staticmethod
    def clean_and_tokenize(a_string: str) -> List[str]:
        return to_word_list(a_string)

    def var_len(self, a_string: str) -> List[int]:
        """
        Creates a variable length list of integer representations of the words
        contained in the provided string.

        :param a_string: (str) an arbitrary string
        :return:
            list: of integer
        """
        word_list = self.clean_and_tokenize(a_string)
        return self.transform_to_vocab_idx_add_eos(word_list)

    def remove_empty(self, a_list: Iterable[int]) -> Iterator[int]:
        """
        Removes all instances of the empty token from an encoded string.

        :param a_list: (list[int]) encoded string
        :return: (list[int]) the encoded string without empty tokens
        """
        return [idx for idx in a_list if idx != self._vocab.empty_token_index]

    def undo(self, a_list: Iterable[int]) -> str:
        """
        Turns a list of integers bak into a readable string using the vocabulary.
        Filters out '<empty>' padding tokens. The reconstructed string will not
        necessarily be the same as the original string, but will be readable.

        :param a_list: (list[int]) encoded string
        :return: (str) reconstruction of the original string
        """
        word_list = self._vocab.lookup(self.remove_empty(a_list))
        return ' '.join(word_list)

    def undo_list(self, list_of_lists: List[List[int]]) -> List[str]:
        """
        As `undo`, but takes a list of lists on ints.

        :param list_of_lists: (list[list[int]]) list of encoded strings
        :return: (list[string]) list of reconstructed strings
        """
        return [self.undo(xs) for xs in list_of_lists]

    def var_len_no_clean(self, a_string: str) -> List[int]:
        """

        :param a_string: (str) string to encode - already tokenized with whitespace separating each token
        :return: (list[str]) list of indices from the given vocab
        """
        return self.transform_to_vocab_idx_add_eos(a_string.split())


class BaseSerializationProcess(ABC):

    @abstractmethod
    def do_serialization(self):
        pass


def estimate_lines(filename, sample_bytes=100000, sample_ratio=0):
    file_size = os.path.getsize(filename)
    read_bytes = sample_bytes
    if sample_ratio > 0:
        read_bytes = file_size * sample_ratio

    if sample_bytes > 0:
        read_bytes = min(sample_bytes, read_bytes)

    read_bytes = int(min(file_size, read_bytes))
    start_point = int(max(file_size / 2 - read_bytes / 2, 0))
    with open(filename, 'rb') as f:
        f.seek(start_point, 0)
        sample = f.read(read_bytes)
        count = sum(1 for b in sample if b == ord('\n'))

    if read_bytes >= file_size:
        return count

    scale_factor = file_size / read_bytes
    return int(count * scale_factor)


class ReadProcess(object):

    def __init__(self):
        self.queue = mp.Queue()

    def fill_queue(self, filename: str, limit: int, verbose: bool):
        """
        Fill the read queue with chat file content.

        :param filename:
        :param limit:
        :param verbose:
        :return: number of conversations written in the queue
        """
        lines_1perc = max(int(estimate_lines(filename) / 100), 1)
        conv_number = 0
        with open(filename) as f:
            for interaction in f:
                self.queue.put(interaction)
                if verbose and conv_number % lines_1perc == 0:
                    try:
                        waiting_read = self.queue.qsize()
                    except NotImplementedError:
                        # According to `multiprocessing.queues.Queue#qsize` `queue.qsize` breaks for OS X
                        waiting_read = conv_number

                    processed = int((conv_number - waiting_read) / lines_1perc)
                    sys.stdout.write('Read: ~{}%, Processed: ~{}%\r'.format(conv_number / lines_1perc, processed))
                    sys.stdout.flush()

                conv_number += 1
                if limit != 0 and conv_number > limit:
                    break

        return conv_number


class Split(Enum):

    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'

    @staticmethod
    def get_split(test_ratio: float, val_ratio: float):
        """

        :param test_ratio:
        :param val_ratio:
        :return:
        """
        v = random.uniform(0, 1)
        if v < test_ratio:
            return Split.TEST
        elif v < (test_ratio + val_ratio):
            return Split.VAL
        else:
            return Split.TRAIN

    @staticmethod
    def to_array():
        """

        :return: an array of the splits
        """
        return [Split.TEST, Split.VAL, Split.TRAIN]


class WriteProcess(object):

    def __init__(self, max_test_conv: int, test_file_folder: str, max_conv_len: int):
        self.queues = {}
        self.processes = []
        self.test_queue = mp.Queue()
        self.max_test_conv = max_test_conv
        self.test_file_folder = test_file_folder
        self.max_conv_len = max_conv_len

    @staticmethod
    def write_files(idx: int, qu: mp.Queue, output_dir: str, finished_processing: mp.Event):
        """Fill the write queues from the read queue"""
        filename = 'conversations{}.tfrecords'.format(idx)
        with tf.python_io.TFRecordWriter(os.path.join(output_dir, filename)) as writer:
            while not finished_processing.is_set() or not qu.empty():
                try:
                    while True:
                        writer.write(qu.get(True, 2))
                except queue.Empty:
                    time.sleep(2)

    def create_folder_if_not_exists(self, output_dir: str, split: Split):
        """Create a folder (if it doesn't already exist) to write split files"""
        self.queues[split.value] = []
        folder = os.path.join(output_dir, split.value)
        if not os.path.exists(folder):
            print('Making new folder:', folder)
            os.makedirs(folder)

        return folder

    def start_filling_queues(self, finished_processing: mp.Event, split: Split, folder: str):
        """Start the processes to fill queues"""
        for x in range(self.max_conv_len):
            qu = mp.Queue()
            self.queues[split.value].append(qu)
            args = (self, x, qu, folder, finished_processing)
            proc = mp.Process(target=self.write_files, args=args)
            proc.start()
            self.processes.append(proc)

    def write_test(self, finished_test_write: mp.Event, finished_write: mp.Event):
        """Write conversation on test file"""
        counter = 0
        with open(self.test_file_folder, 'w') as f:
            while (not finished_write.is_set() or not self.test_queue.empty()) and counter < self.max_test_conv:
                try:
                    while counter < self.max_test_conv:
                        line = self.test_queue.get(True, 2)
                        f.write(','.join([str(x) for x in line]))
                        f.write('\n')
                        counter += 1
                except queue.Empty:
                    time.sleep(2)

        finished_test_write.set()

    def start_write_test(self, finished_test_write: mp.Event, finished_write: mp.Event):
        """Start the process to write tests"""
        args = (finished_test_write, finished_write)
        proc = mp.Process(target=self.write_test, args=args)
        proc.start()
        return proc


class SerializationProcess(BaseSerializationProcess, ABC):

    def __init__(self, encoder: StringEncoder, write_process: WriteProcess,
                 read_process: ReadProcess, n_procs: int, test_ratio: float,
                 val_ratio: float, output_dir: str, filename: str, limit: int):
        self.encoder = encoder
        self.read_process = read_process
        self.write_process = write_process
        self.n_procs = n_procs
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.output_dir = output_dir
        self.filename = filename
        self.limit = limit
        super().__init__()

    @abstractmethod
    def process_line(self, text: str, finished_test_writing: mp.Event):
        """Process a single line"""
        pass

    def serialize(self, finished_reading: mp.Event, finished_test_writing: mp.Event):
        while not finished_reading.is_set() or not self.read_process.queue.empty():
            try:
                while True:
                    self.process_line(self.read_process.queue.get(True, 2), finished_test_writing)
            except queue.Empty:
                time.sleep(2)

    def start_serialization(self, finished_reading: mp.Event, finished_test_writing: mp.Event):
        procs = []
        for i in range(self.n_procs):
            args = (finished_reading, finished_test_writing)
            proc = mp.Process(target=self.serialize, args=args)
            proc.start()
            procs.append(proc)

        return procs

    def do_serialization(self):
        # Multiprocessing events
        finished_processing = mp.Event()
        finished_reading = mp.Event()
        finished_writing = mp.Event()
        finished_test_writing = mp.Event()

        # write out the raw texts included in the test tf-records for later use
        test_proc = self.write_process.start_write_test(finished_test_writing, finished_writing)

        print('Start background write processes')
        for split in Split.to_array():
            folder = self.write_process.create_folder_if_not_exists(self.output_dir, split)
            self.write_process.start_filling_queues(finished_processing, split, folder)

        print('Start background processing')
        procs = self.start_serialization(finished_reading, finished_test_writing)

        print('Reading')
        conv_number = self.read_process.fill_queue(filename=self.filename, limit=self.limit, verbose=True)

        print('Trigger finished-reading-event')
        finished_reading.set()

        # Using the exact number of lines read
        lines_1perc = max(int(conv_number / 100), 1)

        while not self.read_process.queue.empty():
            try:
                waiting_read = self.read_process.queue.qsize()
            except NotImplementedError:
                # According to `multiprocessing.queues.Queue#qsize` `queue.qsize` breaks for OS X
                waiting_read = conv_number

            n_processed = int((conv_number - waiting_read) / lines_1perc)
            sys.stdout.write('Read: 100%, Processed: ~{}%\r'.format(n_processed))
            sys.stdout.flush()
            time.sleep(3)

        print('Waiting for all processes to complete...')
        for proc in procs:
            proc.join()

        print('Trigger finished-processing-event')
        finished_processing.set()

        print('Waiting for all writing to complete...')
        for proc in self.write_process.processes:
            proc.join()

        print('Trigger finished-writing-event')
        finished_writing.set()

        print('Waiting for all writing to test file is complete...')
        test_proc.join()

        self.finish_serialization()
        print('Done')

    @abstractmethod
    def finish_serialization(self):
        """
        Do any final steps required by this instance of the serialization process
        such as writing a summary file
        """
        pass


def hash_text(text: str, encoding: str = 'utf-8') -> str:
    return hashlib.md5(text.encode(encoding)).hexdigest()


def alpha_only(text: str) -> str:
    return re.sub(r'[^a-z]', '', process_response(text).lower())


class ClusterData(object):

    def __init__(self, cluster_filename: str) -> None:
        self.cluster_filename = cluster_filename
        self._data = {}
        self._load()

    def _load(self) -> None:
        with open(self.cluster_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                key = hash_text(alpha_only(row[2]))
                val = hash_text(alpha_only(row[0]))
                self._data[key] = val

    def search(self, q: str) -> Union[str, None]:
        hashed = hash_text(alpha_only(q))
        if hashed in self._data.keys():
            return self._data[hashed]

        return None

    def __len__(self):
        return len(self._data)


class CannedData(object):

    def __init__(self, canned_filename: str) -> None:
        self.canned_filename = canned_filename
        self._data = {}
        self.count = defaultdict(lambda: 0)
        self._load()

    def _load(self) -> None:
        with open(self.canned_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                key = hash_text(alpha_only(row[1]))
                val = row[0]
                self._data[key] = val

    def search_by_key(self, key: str) -> Union[str, None]:
        if key in self._data.keys():
            self.count[key] += 1
            return self._data[key]

        return None

    def __len__(self):
        return len(self._data)


class ReplaceOperator(object):

    def __init__(self, cluster_filename: str, canned_filename: str) -> None:
        self.cluster_data = ClusterData(cluster_filename)
        self.canned_data = CannedData(canned_filename)

    def replace(self, utter: str) -> Union[None, str]:
        try:
            found = self.cluster_data.search(utter)
            if found:
                return self.canned_data.search_by_key(found)

        except AttributeError as e:
            print(e)

        return None


class LpTransform(object):
    """Transforms pseudo JSON file to tf-records (and back) from LP source"""

    class Keys(object):

        UTTERANCES = 'utterances'
        LABEL = 'label'
        UTTER_LENGTHS = 'utter_lengths'
        LABEL_LENGTH = 'label_length'
        ENGAGEMENT_ID = 'engagement_id'
        SKILL_NAMES = 'skill_names'

    def __init__(self, encoder: StringEncoder = None, replace_canned_response: bool = False,
                 filter_canned_response: bool = False):
        self.encoder = encoder
        self.replace_canned_response = replace_canned_response
        self.filter_canned_response = filter_canned_response
        if replace_canned_response or filter_canned_response:
            self.replace_op = ReplaceOperator()

    def json_to_tfrecord(self, seq: List[List[int]], obj: Dict[str, Any], skills: List[int], idx=None):
        if len(seq) == 0:
            raise IndexError('Seq array for context and label definitions must not be empty')

        assert len(seq) == len(skills), 'Length of `seq` and `skills` must match'

        context = seq[:-1]
        label = seq[-1]
        if self.replace_canned_response or self.filter_canned_response:
            label_str = obj['utterances'][idx]['text']
            label_str = self.replace_op.replace(label_str)
            if label_str:
                label = self.encoder.var_len(label_str)
            elif self.filter_canned_response:
                return None

        engagement_id = obj.get('engagementId', -1)
        example = tf.train.SequenceExample()
        example.context.feature[self.Keys.LABEL].int64_list.value.extend(label)
        example.context.feature[self.Keys.LABEL_LENGTH].int64_list.value.append(len(label))
        example.context.feature[self.Keys.ENGAGEMENT_ID].int64_list.value.append(engagement_id)
        example.context.feature[self.Keys.SKILL_NAMES].int64_list.value.extend(skills)
        utterances = example.feature_lists.feature_list(self.Keys.UTTERANCES)
        utter_lengths = example.context.feature[self.Keys.UTTER_LENGTHS]
        for utterance in context:
            utter_lengths.int64_list.value.append(len(utterance))
            utterances.feature.add().int64_list.value.extend(utterance)

        return example.SerializeToString()

    def tfrecord_to_dict(self, record, max_utter_len: int, conv_len: int) -> Dict[str, tf.Tensor]:
        context_features = {
            self.Keys.LABEL: tf.VarLenFeature(tf.int64),
            self.Keys.SKILL_NAMES: tf.VarLenFeature(tf.int64),
            self.Keys.LABEL_LENGTH: tf.FixedLenFeature([], dtype=tf.int64),
            self.Keys.ENGAGEMENT_ID: tf.FixedLenFeature([conv_len], dtype=tf.int64)
        }
        sequence_features = {self.Keys.UTTERANCES: tf.VarLenFeature(tf.int64)}
        features = tf.parse_single_sequence_example(serialized=record, context_features=context_features,
                                                    sequence_features=sequence_features)




class LpSerializationProcess(SerializationProcess):

    def __init__(self, encoder: StringEncoder, full_conv_only: bool, write_process: WriteProcess,
                 read_process: ReadProcess, n_procs: int, replace_canned_response: bool,
                 filter_canned_response: bool, test_ratio: float, val_ratio: float, output_dir: str,
                 filename: str, limit: int):
        super().__init__(encoder, write_process, read_process, n_procs, test_ratio, val_ratio, output_dir, limit)
        self.output_dir = output_dir
        self.replace_canned_response = replace_canned_response
        self.filter_canned_response = filter_canned_response
        self.full_conv_only = full_conv_only
        self.transform =