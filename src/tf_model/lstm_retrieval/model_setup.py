import numpy as np
import os
import tensorflow as tf
import typing
from typing import Any, List, Optional


class _Labels(object):

    GLOBAL_STEP_LABEL = 'global_step'

    OPTIMIZATION_OP_LABEL = 'train_and_update'

    LOSS_LABEL = 'loss'

    SET_TRAINING_LABEL = 'set_training'

    IS_TRAINING_LABEL = 'is_training'

    READ_TRAINING_LABEL = 'read_training'


class BaseModelGraphOps(object):

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    @classmethod
    def load_saved_model_graph(cls, model_meta_path: str) -> tf.train.Saver:
        print('Loading saved model graph and variables from', model_meta_path)
        return tf.train.import_meta_graph(model_meta_path)

    @classmethod
    def reconstruct_saved_model_variables(cls, sess: tf.Session, saver: tf.train.Saver, model_dir: str) -> None:
        print('Loading saved model variables from', model_dir)
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, latest_checkpoint)

    @classmethod
    def initialize_model_graph(cls, sess: tf.Session) -> None:
        """Hook to run any initializations such as setting lookup tables"""
        sess.run(cls.get_init_ops())

    @classmethod
    def signature_def_map(cls, graph: tf.Graph) -> dict:
        """
        Note: this must be implemented to export a saved model.

        :param graph:
        :return: the `signature_def_map` used by TensorFlow Serving
        """
        raise NotImplementedError()

    @classmethod
    def get_init_ops(cls) -> tf.Operation:
        """Hook to get all initializations such as to setup a lookup table"""
        return tf.group(tf.tables_initializer(), name='legacy_init_op')

    @classmethod
    def get_is_training(cls, graph: tf.Graph) -> tf.Tensor:
        return graph.get_tensor_by_name('{}:0'.format(_Labels.IS_TRAINING_LABEL))

    @classmethod
    def get_set_training(cls, graph: tf.Graph) -> tf.Tensor:
        return graph.get_tensor_by_name('{}:0'.format(_Labels.SET_TRAINING_LABEL))

    @classmethod
    def get_read_training(cls, graph: tf.Graph) -> tf.Tensor:
        return graph.get_tensor_by_name('{}:0'.format(_Labels.READ_TRAINING_LABEL))

    @classmethod
    def set_training_mode(cls, sess: tf.Session, is_training: bool) -> None:
        sess.run(cls.get_set_training(sess.graph), {cls.get_read_training(sess.graph): is_training})

    @classmethod
    def get_global_step(cls, graph: tf.Graph) -> tf.Tensor:
        return graph.get_tensor_by_name('{}:0'.format(_Labels.GLOBAL_STEP_LABEL))

    @classmethod
    def get_optimization_op(cls, graph: tf.Graph) -> tf.Operation:
        return graph.get_operation_by_name(_Labels.OPTIMIZATION_OP_LABEL)

    def get_loss_tensor(self, graph: tf.Graph) -> tf.Tensor:
        """
        Note: if you give your loss component a different name, you must overwrite this method.

        :param graph:
        :return:
        """
        return graph.get_tensor_by_name('{}:0'.format(_Labels.LOSS_LABEL))

    def get_metrics(self, graph: tf.Graph) -> dict:
        """

        :param graph:
        :return: keys are the labels, values are the graph components
        """
        return {'loss': self.get_loss_tensor(graph)}


class BaseModelGraphConstructor(object):
    """
    Abstract class for models. Used by ModelTrainer.
    """
    T = typing.TypeVar('T')

    def __init__(self, model_name: str, model_ops: BaseModelGraphOps):
        self.model_name = model_name
        self.model_ops = model_ops
        self.train_metrics_prefix = 'train_'
        self.val_metrics_prefix = 'val_'

    def prepare_trainval_datasets(self, sess: tf.Session, batch_size: int, current_step: int, force_reload_data: bool):
        """Hook to load and prepare training and validation datasets"""
        pass

    def get_next_training_feed_dict(self, sess: tf.Session) -> dict:
        """

        :param sess:
        :return: (dict) `feed_dict` used in `sess.run()`
        """
        raise NotImplementedError

    def get_next_validation_feed_dict(self, sess: tf.Session) -> dict:
        """

        :param sess:
        :return: (dict) `feed_dict` used in `sess.run()`
        """
        raise NotImplementedError

    def custom_action(self, sess: tf.Session, feed_dict: dict) -> None:
        """Hook to do custom stuff such as printing out something every `custom_action_every_k_steps`"""
        pass

    def custom_validation_action(self, sess: tf.Session, feed_dicts: List[dict]) -> None:
        """Hook to do custom stuff such as printing out something when running the validation step"""
        pass

    def pre_session_hook(self) -> Any:
        """
        In case you need to do something before the session is created.

        Can return any object, which will be passed into `Model().construct_full_training_graph()`.
        :return:
        """
        pass

    def construct_model_graph_components(self, graph: tf.Graph, pre_session_obj: T):
        """

        :param graph:
        :param pre_session_obj: whatever was returned from `pre_session_hook`
        :return:
        """
        raise NotImplementedError()

    def construct_is_training_graph_component(self):
        is_training = tf.Variable(True, trainable=False, name=_Labels.IS_TRAINING_LABEL)
        read_training = tf.placeholder(tf.bool, shape=[], name=_Labels.READ_TRAINING_LABEL)
        tf.assign(is_training, read_training, name=_Labels.SET_TRAINING_LABEL)

    def with_clipped_gradients(self, graph: tf.Graph, optimizer, global_step, clip_norm,
                               name_scope='clip_gradients') -> tf.Operation:
        with tf.name_scope(name_scope):
            trainable_vars = tf.trainable_variables()
            grads = tf.gradients(self.model_ops.get_loss_tensor(graph), trainable_vars)
            clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm)

        return optimizer.apply_gradients(zip(clipped_grads, trainable_vars), global_step)

    def construct_full_training_graph_and_init_vars(self, sess: tf.Session, pre_session_obj,
                                                    learning_rate: float,
                                                    clip_gradients_norm: Optional[str]) -> None:
        """

        :param sess:
        :param pre_session_obj:
        :param learning_rate:
        :param clip_gradients_norm:
        :return: the global step variable and the optimization op
        """
        print('Constructing model graph from scratch')
        graph = sess.graph
        global_step = tf.Variable(0, name=_Labels.GLOBAL_STEP_LABEL, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.construct_is_training_graph_component()
        self.construct_model_graph_components(graph, pre_session_obj)
        if clip_gradients_norm:
            optimization_op = self.with_clipped_gradients(graph, optimizer, global_step, clip_gradients_norm)
        else:
            optimization_op = optimizer.minimize(self.model_ops.get_loss_tensor(graph), global_step=global_step)

        # TODO - is the following doing anything?
        update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        with tf.control_dependencies([update_ops]):
            tf.group(optimization_op, name=_Labels.OPTIMIZATION_OP_LABEL)

        sess.run(tf.global_variables_initializer())


class ResponseRetrievalModelGraphConstructor(BaseModelGraphConstructor):

    def __init__(self, model_name: str, model_ops: BaseModelGraphOps, max_num_last_utters: int,
                 embedding: np.ndarray, n_vocab_words: int, data_dir: str, max_shuffle: int,
                 max_utter_len: int, min_conv_len: int, max_conv_len: int, min_remove: int,
                 n_historical_distractors: int, top_k_accuracy: int):
        super().__init__(model_name=model_name, model_ops=model_ops)
        self.max_num_last_utters = max_num_last_utters
        self.embedding = embedding
        self.n_vocab_words = n_vocab_words
        self.data_dir = data_dir
        self.max_shuffle = max_shuffle
        self.max_utter_len = max_utter_len
        self.min_conv_len = min_conv_len
        self.max_conv_len = max_conv_len
        self.min_remove = min_remove
        self.n_historical_distractors = n_historical_distractors
        self.top_k_accuracy = top_k_accuracy

        # Need a session to create these; done inside `initialize`
        self._training_dataset = None
        self._val_dataset = None

    def prepare_trainval_datasets(self, sess: tf.Session, batch_size: int, current_step: int, force_reload_data: bool):
        if current_step != 0 and not force_reload_data:
            return

        def _load(sub_dir: str, _batch_size: int):
            folder = os.path.join(self.data_dir, sub_dir)
            decoder = 


class ModelGraphConstructor(ResponseRetrievalModelGraphConstructor):
    pass
