from collections import defaultdict
import keras
import keras.backend as ke
from keras.models import save_model
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class TQDMProgressCallback(keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.epochs = 0
        self.use_steps = False
        self.target = None
        self.progress_bar = None
        self.log_values_by_metric = None

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d' % (epoch + 1, self.epochs))
        if "steps" in self.params:
            self.use_steps = True
            self.target = self.params['steps']
        else:
            self.use_steps = False
            self.target = self.params['samples']
        self.progress_bar = tqdm(total=self.target)
        self.log_values_by_metric = defaultdict(list)

    def _set_progress_bar_desc(self, logs):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values_by_metric[k].append(logs[k])
        desc = "; ".join("{0}: {1:.4f}".format(k, np.mean(values)) for k, values in self.log_values_by_metric.items())
        if hasattr(self.progress_bar, "set_description_str"):  # for new tqdm versions
            self.progress_bar.set_description_str(desc)
        else:
            self.progress_bar.set_description(desc)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self.use_steps:
            self.progress_bar.update(1)
        else:
            batch_size = logs.get('size', 0)
            self.progress_bar.update(batch_size)
        self._set_progress_bar_desc(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._set_progress_bar_desc(logs)
        self.progress_bar.update(1)  # workaround to show description
        self.progress_bar.close()


class ModelSaveCallback(keras.callbacks.Callback):

    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        model_filename = self.file_name.format(epoch)
        save_model(self.model, model_filename)
        print("Model saved in {}".format(model_filename))


# remember to clear session/graph if you rebuild your graph to
# avoid out-of-memory errors
def reset_tf_session():
    ke.clear_session()
    tf.reset_default_graph()
    sess = ke.get_session()
    return sess
