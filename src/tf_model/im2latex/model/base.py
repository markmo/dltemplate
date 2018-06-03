import sys
import tensorflow as tf
from tf_model.im2latex.utils.general import get_logger, init_dir
import time


class BaseModel(object):
    """
    Generic class for tf models
    """

    def __init__(self, config, dir_output):
        self._config = config
        self._dir_output = dir_output
        init_dir(dir_output)
        self.logger = get_logger(self._dir_output + 'model.log')
        tf.reset_default_graph()  # safeguard if previous model was defined
        self.sess = None
        self.saver = None

    def build_train(self, config=None):
        """
        To overwrite with model-specific logic

        This logic must define:
        * loss
        * lr
        * etc.

        :param config:
        :return:
        """
        raise NotImplementedError

    def build_pred(self, config=None):
        """
        Similar to `build_train` but no need to define `train_op`
        :param config:
        :return:
        """
        raise NotImplementedError

    def _add_train_op(self, lr_method, lr, loss, clip=-1):
        """
        Defines `train_op` that performs an update on a batch

        :param lr_method: (string) sgd method, e.g. 'adam'
        :param lr:  (tf.placeholder) tf.float32, learning rate
        :param loss: (tensor) tf.float32, loss to minimize
        :param clip: (python float) clipping of gradient.
                     If clip <= 0, then no clipping
        :return:
        """
        _lr_m = lr_method.lower()
        with tf.variable_scope('train_step'):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError('Unknown method {}'.format(_lr_m))

            # for batch norm beta gamma
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if clip > 0:  # gradient clipping if clip is positive
                    grads, vs = zip(*optimizer.compute_gradients(loss))
                    grads, _ = tf.clip_by_global_norm(grads, clip)
                    self.train_op = optimizer.apply_gradients(zip(grads, vs))
                else:
                    self.train_op = optimizer.minimize(loss)

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        """
        Reload weights into session

        :param dir_model: dir with weights
        :return:
        """
        self.logger.info('Reloading the latest trained model...')
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        """
        Saves session

        :return:
        """
        # check dir one last time
        dir_model = self._dir_output + 'model.weights/'
        init_dir(dir_model)

        # logging
        sys.stdout.write('\r- Saving model...')
        sys.stdout.flush()

        # saving
        self.saver.save(self.sess, dir_model)

        # logging
        sys.stdout.write('\r')
        sys.stdout.flush()
        self.logger.info('- Saved model in {}'.format(dir_model))

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self._dir_output, self.sess.graph)

    def train(self, config, train_set, val_set, lr_schedule):
        """
        Global training procedure

        Call `run_epoch` and saves weights if score improves. All the
        epoch-logic including the `lr_schedule` update must be done
        in `run_epoch`.

        :param config: Config instance contains params as attributes
        :param train_set: Dataset instance
        :param val_set: Dataset instance
        :param lr_schedule: LRSchedule instance that takes care of learning
        :return: best_score (float)
        """
        best_score = None
        for epoch in range(config.n_epochs):
            # logging
            tic = time.time()
            self.logger.info('Epoch {:}/{:}'.format(epoch + 1, config.n_epochs))

            score = self._run_epoch(config, train_set, val_set, epoch, lr_schedule)

            # save weights if we have new best score on eval
            if best_score is None or score > best_score:
                best_score = score
                self.logger.info('- New best score ({:04.2f})!'.format(best_score))
                self.save_session()

            if lr_schedule.stop_training:
                self.logger.info('- Early stopping')
                break

            # logging
            toc = time.time()
            self.logger.info('- Elapsed time: {:04.2f}, lr: {:04.5f}'.format(toc - tic, lr_schedule.lr))

        return best_score

    def _run_epoch(self, config, train_set, val_set, epoch, lr_schedule):
        """
        Performs an epoch of training

        Model specific method to overwrite

        :param config:
        :param train_set:
        :param val_set:
        :param epoch: (int) epoch id, starting at 0
        :param lr_schedule: (LRSchedule) instance that takes care of learning
        :return: (float) model will select weights that achieve the
                 highest score
        """
        raise NotImplementedError

    def evaluate(self, config, test_set):
        """
        Evaluates model on test set

        Calls method `run_evaluate` on test set and takes care of logging

        :param config:
        :param test_set:
        :return: (dict) scores['acc'] = 0.85 for instance
        """
        # logging
        sys.stdout.write('\r- Evaluating...')
        sys.stdout.flush()

        # evaluate
        scores = self._run_evaluate(config, test_set)

        # logging
        sys.stdout.write('\r')
        sys.stdout.flush()
        msg = ' - '.join(['{} {:04.2f}'.format(k, v) for k, v in scores.items()])
        self.logger.nfo('- Eval: {}'.format(msg))

        return scores

    def _run_evaluate(self, config, test_set):
        """
        Performs an epoch of evaluation

        Model-specific method to overwrite

        :param config:
        :param test_set:
        :return: (dict) scores['acc'] = 0.85 for instance
        """
        raise NotImplementedError
