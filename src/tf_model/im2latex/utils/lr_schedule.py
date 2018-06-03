import numpy as np


class LRSchedule(object):
    """
    Class for learning rate schedules

    Implements
    * (time) exponential decay with custom range
        * needs to set `start_decay`, `end_decay`, `lr_init` and `lr_min`
        * set `end_decay` to None to deactivate
    * (time) warm start
        * needs to set `lr_warm`, `end_warm`
        * set `end_warm` to None to deactivate
    * (score) multiplicative decay if no improvement over score
        * needs to set `decay_rate`
        * set `decay_rate` to None to deactivate
    * (score) early stopping if no improvement
        * needs to set `early_stopping`
        * set `early_stopping` to None to deactivate

    All durations are measured in number of batches. For usage, must call
    the update function at each batch. You can access the current learning
    rate with `lr`.
    """

    def __init__(self, lr_init=1e-3, lr_min=1e-4, start_decay=0,
                 decay_rate=None, end_decay=None, lr_warm=1e-4,
                 end_warm=None, early_stopping=None):
        """
        Initializes learning rate schedule.

        Sets `lr` and `stop_training`.

        :param lr_init: (float) initial lr
        :param lr_min: (float)
        :param start_decay: (int) id of batch to start decay
        :param decay_rate: (float) lr *= decay_rate if no improvement.
                           If None, then no multiplicative decay.
        :param end_decay: (int) id of batch to end decay.
                          If None, then no exp decay.
        :param lr_warm: (float) constant learning rate at the beginning
        :param end_warm: (int) id of batch to keep the lr_warm before returning
                         to lr_init and starting the regular schedule.
        :param early_stopping: (int) number of batches with no improvement
        """
        self._lr_init = lr_init
        self._lr_min = lr_min
        self._start_decay = start_decay
        self._decay_rate = decay_rate
        self._end_decay = end_decay
        self._lr_warm = lr_warm
        self._end_warm = end_warm
        self._score = None
        self._early_stopping = early_stopping
        self._n_batch_no_imprv = 0

        # warm start initializes learning rate to warm start
        if self._end_warm is not None:
            # make sure that decay happens after the warm up
            self._start_decay = max(self._end_warm, self._start_decay)
            self.lr = self._lr_warm
        else:
            self.lr = lr_init

        # setup of exponential decay
        if self._end_decay is not None:
            self._exp_decay = np.power(lr_min / lr_init, 1 / float(self._end_decay - self._start_decay))

    @property
    def stop_training(self):
        """For early stopping"""
        return self._early_stopping is not None and self._n_batch_no_imprv >= self._early_stopping

    def update(self, batch_no=None, score=None):
        """
        Updates the learning rate.

        Decay by decay_rate if score is higher than previous.
        Update lr according to warm up and exp decay. Both updates can occur
        concurrently.

        :param batch_no: (int) batch id
        :param score: (float) score, higher is better
        :return:
        """
        # update based on time
        if batch_no is not None:
            if self._end_warm is not None and self._end_warm <= batch_no <= self._start_decay:
                self.lr = self._lr_init

            if batch_no > self._start_decay and self._end_decay is not None:
                self.lr *= self._exp_decay

        # update based on performance
        if self._decay_rate is not None:
            if score is not None and self._score is not None:
                if score <= self._score:
                    self.lr *= self._decay_rate
                    self._n_batch_no_imprv += 1
                else:
                    self._n_batch_no_imprv = 0

        # update last score eval
        if score is not None:
            self._score = score

        self.lr = max(self.lr, self._lr_min)
