from collections import defaultdict
from functools import reduce
import os
from tensorflow.contrib.legacy_seq2seq import sequence_loss
from tf_model.neural_turing_machine.ops import *
from tf_model.neural_turing_machine.util import *


class NTMCell(object):

    def __init__(self, input_dim, output_dim,
                 mem_size=128, mem_dim=20, controller_dim=100,
                 controller_layer_size=1, shift_range=1,
                 write_head_size=1, read_head_size=1):
        """
        Initialize the parameters for an NTM cell.

        :param input_dim: (int) number of units in the LSTM cell
        :param output_dim: (int) dimensionality of the inputs into the LSTM cell
        :param mem_size: (int) (optional) size of memory
        :param mem_dim: (int) (optional) dimensionality for memory
        :param controller_dim: (int) (optional) dimensionality for controller
        :param controller_layer_size: (int) (optional) size of controller layer
        :param shift_range:
        :param write_head_size:
        :param read_head_size:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller_dim = controller_dim
        self.controller_layer_size = controller_layer_size
        self.shift_range = shift_range
        self.write_head_size = write_head_size
        self.read_head_size = read_head_size
        self.depth = 0
        self.states = []

    def __call__(self, input_, state=None, scope=None):
        """
        Run one step of the NTM.

        :param input_: (2D Tensor) 1 x input_size
        :param state: (dict) {M, read_w, write_w, read, output, hidden}
        :param scope: VariableScope for the created subgraph; defaults to class name.
        :return: A tuple:
                 - A 2D, batch x output_dim, Tensor representing the output of the LSTM
                   after reading "input_" when previous state was "state".
                   Here output_dim is:
                       num_proj if num_proj was set,
                       num_units otherwise
                 - A 2D, batch x state_size, Tensor representing the new state of LSTM
                   after reading "input_" when previous state was "state".
        """
        if state is None:
            _, state = self.initial_state()

        m_prev = state['M']
        read_w_list_prev = state['read_w']
        write_w_list_prev = state['write_w']
        read_list_prev = state['read']
        output_list_prev = state['output']
        hidden_list_prev = state['hidden']

        # build a controller
        output_list, hidden_list = self.build_controller(input_, read_list_prev, output_list_prev, hidden_list_prev)

        # last output layer from LSTM controller
        last_output = output_list[-1]

        # build a memory
        m, read_w_list, write_w_list, read_list = self.build_memory(m_prev,
                                                                    read_w_list_prev,
                                                                    write_w_list_prev,
                                                                    last_output)

        # get a new output
        new_output, new_output_logit = self.new_output(last_output)

        state = {
            'M': m,
            'read_w': read_w_list,
            'write_w': write_w_list,
            'read': read_list,
            'output': output_list,
            'hidden': hidden_list
        }
        self.depth += 1
        self.states.append(state)

        return new_output, new_output_logit, state

    def new_output(self, output):
        """ Logistic sigmoid output layers """
        with tf.variable_scope('output'):
            logit = Linear(output, self.output_dim, name='output')
            return tf.sigmoid(logit), logit

    def build_controller(self, input_, read_list_prev, output_list_prev, hidden_list_prev):
        """ Build LSTM controller """
        with tf.variable_scope('controller'):
            output_list = []
            hidden_list = []
            for layer_idx in range(self.write_head_size):
                o_prev = output_list_prev[layer_idx]
                h_prev = hidden_list_prev[layer_idx]
                if layer_idx == 0:
                    def new_gate(gate_name):
                        return linear([input_, o_prev] + read_list_prev,
                                      output_size=self.controller_dim,
                                      bias=True,
                                      scope='%s_gate_%s' % (gate_name, layer_idx))
                else:
                    def new_gate(gate_name):
                        return linear([output_list[-1], o_prev],
                                      output_size=self.controller_dim,
                                      bias=True,
                                      scope='%s_gate_%s' % (gate_name, layer_idx))

                # input, forget, output, and update gates for LSTM
                input_gate = tf.sigmoid(new_gate('input'))
                forget_gate = tf.sigmoid(new_gate('forget'))
                output_gate = tf.sigmoid(new_gate('output'))
                update_gate = tf.tanh(new_gate('update'))

                # update the state of the LSTM cell
                hidden = tf.add_n([forget_gate * h_prev, input_gate * update_gate])
                output = output_gate * tf.tanh(hidden)

                hidden_list.append(hidden)
                output_list.append(output)

            return output_list, hidden_list

    def build_memory(self, m_prev, read_w_list_prev, write_w_list_prev, last_output):
        """ Build a memory to read and write """
        with tf.variable_scope('memory'):
            # Reading
            if self.read_head_size == 1:
                read_w_prev = read_w_list_prev[0]
                read_w, read = self.build_read_head(m_prev, tf.squeeze(read_w_prev), last_output, 0)
                read_w_list = [read_w]
                read_list = [read]
            else:
                read_w_list = []
                read_list = []

                for idx in range(self.read_head_size):
                    read_w_prev_idx = read_w_list_prev[idx]
                    read_w_idx, read_idx = self.build_read_head(m_prev, read_w_prev_idx, last_output, idx)
                    read_w_list.append(read_w_idx)
                    read_list.append(read_idx)

            # Writing
            if self.write_head_size == 1:
                write_w_prev = write_w_list_prev[0]
                write_w, write, erase = self.build_write_head(m_prev, tf.squeeze(write_w_prev), last_output, 0)
                m_erase = tf.ones([self.mem_size, self.mem_dim]) - outer_product(write_w, erase)
                m_write = outer_product(write_w, write)
                write_w_list = [write_w]
            else:
                write_w_list = []
                write_list = []
                erase_list = []
                m_erases = []
                m_writes = []
                for idx in range(self.write_head_size):
                    write_w_prev_idx = write_w_list_prev[idx]
                    write_w_idx, write_idx, erase_idx = self.build_write_head(m_prev, write_w_prev_idx,
                                                                              last_output, idx)
                    write_w_list.append(tf.transpose(write_w_idx))
                    write_list.append(write_idx)
                    erase_list.append(erase_idx)
                    m_erases.append(tf.ones([self.mem_size, self.mem_dim]) - outer_product(write_w_idx, erase_idx))
                    m_writes.append(outer_product(write_w_idx, write_idx))

                m_erase = reduce(lambda x, y: x * y, m_erases)
                m_write = tf.add_n(m_writes)

            m = m_prev * m_erase + m_write

            return m, read_w_list, write_w_list, read_list

    def build_read_head(self, m_prev, read_w_prev, last_output, idx):
        return self.build_head(m_prev, read_w_prev, last_output, idx, is_read=True)

    def build_write_head(self, m_prev, write_w_prev, last_output, idx):
        return self.build_head(m_prev, write_w_prev, last_output, idx, is_read=False)

    # noinspection SpellCheckingInspection
    def build_head(self, m_prev, w_prev, last_output, idx, is_read):
        scope = 'read' if is_read else 'write'
        with tf.variable_scope(scope):
            # Amplify or attenuate the precision
            with tf.variable_scope('k'):
                k = tf.tanh(Linear(last_output, self.mem_dim, name='k_%s' % idx))

            # Interpolation gate
            with tf.variable_scope('g'):
                g = tf.sigmoid(Linear(last_output, 1, name='g_%s' % idx))

            # shift weighting
            with tf.variable_scope('s_w'):
                w = Linear(last_output, 2 * self.shift_range + 1, name='s_w_%s' % idx)
                s_w = softmax(w)

            with tf.variable_scope('beta'):
                beta = tf.nn.softplus(Linear(last_output, 1, name='beta_%s' % idx))

            with tf.variable_scope('gamma'):
                gamma = tf.add(tf.nn.softplus(Linear(last_output, 1, name='gamma_%s' % idx)), tf.constant(1.0))

            # Cosine similarity
            similarity = smooth_cosine_similarity(m_prev, k)  # [mem_size x 1]

            # Focusing by content
            content_focused_w = softmax(scalar_mul(similarity, beta))

            # Focusing by location
            gated_w = tf.add_n([scalar_mul(content_focused_w, g), scalar_mul(w_prev, tf.constant(1.0) - g)])

            # Convolutional shifts
            conv_w = circular_convolution(gated_w, s_w)

            # Sharpening
            powed_conv_w = tf.pow(conv_w, gamma)
            w = powed_conv_w / tf.reduce_sum(powed_conv_w)

            if is_read:
                # Reading
                read = matmul(tf.transpose(m_prev), w)
                return w, read
            else:
                # Writing
                erase = tf.sigmoid(Linear(last_output, self.mem_dim, name='erase_%s' % idx))
                add = tf.tanh(Linear(last_output, self.mem_dim, name='add_%s' % idx))
                return w, add, erase

    def initial_state(self, dummy_value=0.0):
        self.depth = 0
        self.states = []
        with tf.variable_scope('init_cell'):
            # always zero
            dummy = tf.Variable(tf.constant([[dummy_value]], dtype=tf.float32))

            # memory
            m_init_linear = tf.tanh(Linear(dummy, self.mem_size * self.mem_dim, name='m_init_linear'))
            m_init = tf.reshape(m_init_linear, [self.mem_size, self.mem_dim])

            # read weights
            read_w_list_init = []
            read_list_init = []
            for idx in range(self.read_head_size):
                read_w_idx = Linear(dummy, self.mem_size, is_range=True, squeeze=True, name='read_w_%d' % idx)
                read_w_list_init.append(softmax(read_w_idx))
                read_init_idx = Linear(dummy, self.mem_dim, squeeze=True, name='read_init_%d' % idx)
                read_list_init.append(tf.tanh(read_init_idx))

            # write weights
            write_w_list_init = []
            for idx in range(self.write_head_size):
                write_w_idx = Linear(dummy, self.mem_size, is_range=True, squeeze=True, name='write_w_%s' % idx)
                write_w_list_init.append(softmax(write_w_idx))

            # controller state
            output_init_list = []
            hidden_init_list = []
            for idx in range(self.controller_layer_size):
                output_init_idx = Linear(dummy, self.controller_dim, squeeze=True, name='output_init_%s' % idx)
                output_init_list.append(tf.tanh(output_init_idx))
                hidden_init_idx = Linear(dummy, self.controller_dim, squeeze=True, name='hidden_init_%s' % idx)
                hidden_init_list.append(tf.tanh(hidden_init_idx))

            output = tf.tanh(Linear(dummy, self.output_dim, name='new_output'))
            state = {
                'M': m_init,
                'read_w': read_w_list_init,
                'write_w': write_w_list_init,
                'read': read_list_init,
                'output': output_init_list,
                'hidden': hidden_init_list
            }
            self.depth += 1
            self.states.append(state)

            return output, state

    def get_memory(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['M']

    def get_read_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read_w']

    def get_write_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['write_w']

    def get_read_vector(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read']

    def print_read_max(self, sess):
        read_w_list = sess.run(self.get_read_weights())
        fmt = '%-4d %.4f'
        if self.read_head_size == 1:
            print(fmt % argmax(read_w_list[0]))
        else:
            for idx in range(self.read_head_size):
                print(fmt % argmax(read_w_list[idx]))

    def print_write_max(self, sess):
        write_w_list = sess.run(self.get_write_weights())
        fmt = '%-4d %.4f'
        if self.write_head_size == 1:
            print(fmt % argmax(write_w_list[0]))
        else:
            for idx in range(self.write_head_size):
                print(fmt % argmax(write_w_list[idx]))


class NTM(object):

    def __init__(self, cell, sess,
                 min_length, max_length, test_max_length=120,
                 min_grad=-10, max_grad=10,
                 lr=1e-4, momentum=0.9, decay=0.95,
                 scope='NTM', forward_only=False):
        """
        Creates a Neural Turing Machine (NTM) specified by cell (NTMCell).

        :param cell: an instance of NTMCell
        :param sess: a TensorFlow session
        :param min_length: minimum length of input sequence
        :param max_length: maximum length of input sequence for training
        :param test_max_length: maximum length of input sequence for testing
        :param min_grad: (optional) minimum gradient for gradient clipping [-10]
        :param max_grad: (optional) maximum gradient for gradient clipping [+10]
        :param lr: (optional) learning rate
        :param momentum: (optional) momentum of RMSProp
        :param decay: (optional) decay rate of RMSProp
        :param scope:
        :param forward_only:
        """
        if not isinstance(cell, NTMCell):
            raise TypeError('cell must be an instance of NTMCell')

        self.cell = cell
        self.sess = sess
        self.scope = scope
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.min_grad = min_grad
        self.max_grad = max_grad
        self.min_length = min_length
        self.max_length = max_length
        self._max_length = max_length

        if forward_only:
            self.max_length = test_max_length

        self.inputs = []
        self.outputs = {}
        self.output_logits = {}
        self.true_outputs = []
        self.prev_states = {}
        self.input_states = defaultdict(list)
        self.output_states = defaultdict(list)
        self.start_symbol = tf.placeholder(tf.float32, [self.cell.input_dim], name='start_symbol')
        self.end_symbol = tf.placeholder(tf.float32, [self.cell.input_dim], name='end_symbol')
        self.losses = {}
        self.optims = {}
        self.grads = {}
        self.saver = None
        self.params = None

        with tf.variable_scope(scope):
            self.global_step = tf.Variable(0, trainable=False)

        self.build_model(forward_only)

    def build_model(self, forward_only, is_copy=True):
        print(' [*] Building an NTM model')

        with tf.variable_scope(self.scope):
            # present start symbol
            prev_state = None
            if is_copy:
                _, _, prev_state = self.cell(self.start_symbol, state=None)
                self.save_state(prev_state, 0, self.max_length)

            zeros = np.zeros(self.cell.input_dim, dtype=np.float32)
            tf.get_variable_scope().reuse_variables()
            for seq_length in range(1, self.max_length + 1):
                progress(seq_length / float(self.max_length))
                input_ = tf.placeholder(tf.float32, [self.cell.input_dim], name='input_%s' % seq_length)
                true_output = tf.placeholder(tf.float32, [self.cell.output_dim], name='true_output_%s' % seq_length)
                self.inputs.append(input_)
                self.true_outputs.append(true_output)

                # present inputs
                _, _, prev_state = self.cell(input_, prev_state)
                self.save_state(prev_state, seq_length, self.max_length)

                # present end symbol
                state = None
                if is_copy:
                    _, _, state = self.cell(self.end_symbol, prev_state)
                    self.save_state(state, seq_length)

                self.prev_states[seq_length] = state

                if not forward_only:
                    # present targets
                    outputs, output_logits = [], []
                    for _ in range(seq_length):
                        output, output_logit, state = self.cell(zeros, state)
                        self.save_state(state, seq_length, is_output=True)
                        outputs.append(output)
                        output_logits.append(output_logit)

                    self.outputs[seq_length] = outputs
                    self.output_logits[seq_length] = output_logits

            if not forward_only:
                for seq_length in range(self.min_length, self.max_length + 1):
                    print(' [*] Building a loss model for seq_length %s' % seq_length)
                    loss = sequence_loss(
                        logits=self.output_logits[seq_length],
                        targets=self.true_outputs[0:seq_length],
                        weights=[1] * seq_length,
                        average_across_timesteps=False,
                        average_across_batch=False,
                        softmax_loss_function=softmax_loss_function
                    )
                    self.losses[seq_length] = loss
                    if not self.params:
                        self.params = tf.trainable_variables()

                    grads = []
                    for grad in tf.gradients(loss, self.params):
                        if grad is not None:
                            grads.append(tf.clip_by_value(grad, self.min_grad, self.max_grad))
                        else:
                            grads.append(grad)

                    self.grads[seq_length] = grads
                    optimizer = tf.train.RMSPropOptimizer(self.lr, decay=self.decay, momentum=self.momentum)
                    reuse = seq_length != 1
                    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                        self.optims[seq_length] = optimizer.apply_gradients(zip(grads, self.params),
                                                                            global_step=self.global_step)

        model_vars = [v for v in tf.global_variables() if v.name.startswith(self.scope)]
        self.saver = tf.train.Saver(model_vars)
        print(' [*] Finished building NTM model')

    def get_outputs(self, seq_length):
        if seq_length not in self.outputs:
            with tf.variable_scope(self.scope):
                tf.get_variable_scope().reuse_variables()
                zeros = np.zeros(self.cell.input_dim, dtype=np.float32)
                state = self.prev_states[seq_length]
                outputs, output_logits = [], []
                for _ in range(seq_length):
                    output, output_logit, state = self.cell(zeros, state)
                    self.save_state(state, seq_length, is_output=True)
                    outputs.append(output)
                    output_logits.append(output_logit)

                self.outputs[seq_length] = outputs
                self.output_logits[seq_length] = output_logits

        return self.outputs[seq_length]

    def get_loss(self, seq_length):
        if seq_length not in self.outputs:
            self.get_outputs(seq_length)

        if seq_length not in self.losses:
            loss = sequence_loss(
                logits=self.output_logits[seq_length],
                targets=self.true_outputs[0:seq_length],
                weights=[1] * seq_length,
                average_across_timesteps=False,
                average_across_batch=False,
                softmax_loss_function=softmax_loss_function
            )
            self.losses[seq_length] = loss

        return self.losses[seq_length]

    def get_output_states(self, seq_length):
        zeros = np.zeros(self.cell.input_dim, dtype=np.float32)
        if seq_length not in self.output_states:
            with tf.variable_scope(self.scope):
                tf.get_variable_scope().reuse_variables()
                outputs, output_logits = [], []
                state = self.prev_states[seq_length]
                for _ in range(seq_length):
                    output, output_logit, state = self.cell(zeros, state)
                    self.save_state(state, seq_length, is_output=True)
                    outputs.append(output)
                    output_logits.append(output_logit)

                self.outputs[seq_length] = outputs
                self.output_logits[seq_length] = output_logits

        return self.output_states[seq_length]

    @property
    def loss(self):
        return self.losses[self.cell.depth]

    @property
    def optimizer(self):
        return self.optims[self.cell.depth]

    def save_state(self, state, from_, to=None, is_output=False):
        if is_output:
            state_to_add = self.output_states
        else:
            state_to_add = self.input_states

        if to:
            for i in range(from_, to + 1):
                state_to_add[i].append(state)
        else:
            state_to_add[from_].append(state)

    def save(self, checkpoint_dir, task_name, step):
        task_dir = os.path.join(checkpoint_dir, '%s_%s' % (task_name, self.max_length))
        filename = '%s_%s.model' % (self.scope, task_name)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        self.saver.save(self.sess, os.path.join(task_dir, filename), global_step=step.astype(int))

    def load(self, checkpoint_dir, task_name, strict=True):
        print(' [*] Reading checkpoints...')
        task_dir = '%s_%s' % (task_name, self._max_length)
        checkpoint_dir = os.path.join(checkpoint_dir, task_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            if strict:
                raise Exception(' [!] Testing, but %s not found' % checkpoint_dir)
            else:
                print(' [!] Training, but previous training data %s not found' % checkpoint_dir)


def softmax_loss_function(labels, inputs):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=inputs)
