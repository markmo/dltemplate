import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import while_loop
from tensorflow.python.ops.tensor_array_ops import TensorArray


class LeakGANModel(object):

    def __init__(self, seq_len, n_classes, vocab_size, embed_dim, dis_embed_dim, filter_sizes,
                 n_filters, batch_size, n_hidden, start_token, goal_out_size, goal_size,
                 step_size, d_model, n_layers=1, l2_reg_lambda=0, learning_rate=0.001):
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dis_embed_dim = dis_embed_dim
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.n_layers = n_layers
        self.l2_reg_lambda = l2_reg_lambda
        self.init_learning_rate = learning_rate
        self.n_filters_sum = sum(n_filters)
        self.grad_clip = 5.0
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size
        self.step_size = step_size
        self.d_model = d_model
        self.feature_extractor_unit = d_model.feature_extractor_unit
        global_step_pre = tf.Variable(0, trainable=False)
        learning_rate_pre = tf.train.exponential_decay(self.init_learning_rate, global_step_pre,
                                                       200, 0.96, staircase=True)
        global_step_adv = tf.Variable(0, trainable=False)
        learning_rate_adv = tf.train.exponential_decay(self.init_learning_rate, global_step_adv,
                                                       30, 0.96, staircase=True)
        self.scope = d_model.feature_scope
        self.worker_params = []
        self.manager_params = []
        self.epis = 0.65
        self.tem = 1.0
        with tf.variable_scope('placeholder'):
            self.x = tf.placeholder(tf.int32, shape=[batch_size, seq_len])  # sequence of tokens from generator
            self.reward = tf.placeholder(tf.float32, shape=[batch_size, seq_len / step_size])
            self.given_num = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.train = tf.placeholder(tf.int32, None, name='train')

        with tf.variable_scope('worker'):
            self.g_embeddings = tf.Variable(tf.random_normal([vocab_size, embed_dim], stddev=0.1))
            self.worker_params.append(self.g_embeddings)
            # maps h_tm1 to h_t for generator
            self.g_worker_recurrent_unit = self.create_worker_recurrent_unit(self.worker_params)
            # maps h_t to o_t (output token logits)
            self.g_worker_output_unit = self.create_worker_output_unit(self.worker_params)
            self.w_worker_out_change = tf.Variable(tf.random_normal([vocab_size, goal_size], stddev=0.1))
            self.g_change = tf.Variable(tf.random_normal([goal_out_size, goal_size], stddev=0.1))
            self.worker_params.extend([self.w_worker_out_change, self.g_change])
            self.h0_worker = tf.zeros([batch_size, n_hidden])
            self.h0_worker = tf.stack([self.h0_worker, self.h0_worker])

        with tf.variable_scope('manager'):
            # maps h_tm1 to h_t for generator
            self.g_manager_recurrent_unit = self.create_manager_recurrent_unit(self.manager_params)
            # maps h_t to o_t (output token logits)
            self.g_manager_output_unit = self.create_manager_output_unit(self.manager_params)
            self.h0_manager = tf.zeros([batch_size, n_hidden])
            self.h0_manager = tf.stack([self.h0_manager, self.h0_manager])
            self.goal_init = tf.get_variable('goal_init',
                                             initializer=tf.truncated_normal([batch_size, goal_out_size], stddev=0.1))
            self.manager_params.extend([self.goal_init])

        self.padding_array = tf.constant(-1, shape=[batch_size, seq_len], dtype=tf.int32)

        with tf.name_scope('rollout'):
            self.gen_for_reward = self.rollout(self.x, self.given_num)

        # processed for batch
        with tf.device('/cpu:0'):
            # shape [seq_len, batch_size, embed_dim]
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])

        gen_o = TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)
        gen_x = TensorArray(dtype=tf.int32, size=1, dynamic_size=True, infer_shape=True, clear_after_read=False)
        goal = TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True, clear_after_read=False)
        feature_array = TensorArray(dtype=tf.float32, size=seq_len + 1, dynamic_size=False, infer_shape=True,
                                    clear_after_read=False)
        real_goal_array = TensorArray(dtype=tf.float32, size=seq_len / step_size, dynamic_size=False,
                                      infer_shape=True, clear_after_read=False)
        gen_real_goal_array = TensorArray(dtype=tf.float32, size=seq_len / step_size, dynamic_size=False,
                                          infer_shape=True, clear_after_read=False)
        gen_o_worker_array = TensorArray(dtype=tf.float32, size=seq_len / step_size, dynamic_size=False,
                                         infer_shape=True, clear_after_read=False)

        def g_recurrence(i, x_t, h_tm1, h_tm1_manager, gen_o, gen_x, goal, last_goal, real_goal, step_size,
                         gen_real_goal_array, gen_o_worker_array):
            # padding sentence by -1
            cur_sent = tf.cond(i > 0,
                               lambda: tf.split(
                                   tf.concat([tf.transpose(gen_x.stack(), perm=[1, 0]), self.padding_array], 1),
                                   [seq_len, i], 1)[0],
                               lambda: self.padding_array)
            with tf.variable_scope(self.scope):
                feature = self.feature_extractor_unit(cur_sent, self.keep_prob)

            h_t_worker = self.g_worker_recurrent_unit(x_t, h_tm1)  # hidden memory tuple
            o_t_worker = self.g_worker_output_unit(h_t_worker)  # shape [batch_size, vocab_size], logits not prob
            o_t_worker = tf.reshape(o_t_worker, [batch_size, vocab_size, goal_size])
            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager)
            sub_goal = self.g_manager_output_unit(h_t_manager)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)
            goal = goal.write(i, sub_goal)
            real_sub_goal = tf.add(last_goal, sub_goal)
            w_g = tf.matmul(real_goal, self.g_change)  # shape [batch_size, goal_size]
            w_g = tf.nn.l2_normalize(w_g, 1)
            gen_real_goal_array = gen_real_goal_array.write(i, real_goal)
            w_g = tf.expand_dims(w_g, 2)  # shape [batch_size, goal_size, 1]
            gen_o_worker_array = gen_o_worker_array.write(i, o_t_worker)
            x_logits = tf.matmul(o_t_worker, w_g)
            x_logits = tf.squeeze(x_logits)
            log_prob = tf.log(tf.nn.softmax(tf.cond(i > 1,
                                                    lambda: tf.cond(self.train > 0, lambda: self.tem, lambda: 1.5),
                                                    lambda: 1.5) * x_logits))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # shape [batch_size, embed_dim]
            with tf.control_dependencies([cur_sent]):
                gen_x = gen_x.write(i, next_token)  # shape [indices, batch_size]

            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, vocab_size, 1.0, 1.0),
                                                             tf.nn.softmax(x_logits)), 1))  # [batch_size], prob

            return (i + 1, x_tp1, h_t_worker, h_t_manager, gen_o, gen_x, goal,
                    tf.cond(((i + 1) % step_size) > 0,
                            lambda: real_sub_goal,
                            lambda: tf.constant(0, shape=[batch_size, goal_out_size])),
                    tf.cond(((i + 1) % step_size) > 0,
                            lambda: real_goal,
                            lambda: real_sub_goal),
                    step_size, gen_real_goal_array, gen_o_worker_array)

        _, _, _, _, self.gen_o, self.gen_x, _, _, _, _, self.gen_real_goal_array, self.gen_o_worker_array = \
            while_loop(cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11: i < seq_len,
                       body=g_recurrence,
                       loop_vars=(tf.constant(0, dtype=tf.int32),
                                  tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                                  self.h0_worker, self.h0_manager,
                                  gen_o, gen_x, goal,
                                  tf.zeros([batch_size, goal_out_size]),
                                  self.goal_init, step_size, gen_real_goal_array, gen_o_worker_array),
                       parallel_iterations=1)
        self.gen_x = self.gen_x.stack()  # shape [seq_len, batch_size]
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # shape [batch_size, seq_len]
        self.gen_real_goal_array = self.gen_real_goal_array.stack()  # shape [seq_len, batch_size, goal_size]
        # shape [batch_size, seq_len, goal_size]
        self.gen_real_goal_array = tf.transpose(self.gen_real_goal_array, perm=[1, 0, 2])
        self.gen_o_worker_array = self.gen_o_worker_array.stack()  # shape [seq_len, batch_size, vocab_size, goal_size]
        # shape [batch_size, seq_len, vocab_size, goal_size]
        self.gen_o_worker_array = tf.transpose(self.gen_o_worker_array, perm=[1, 0, 2, 3])
        sub_feature = TensorArray(dtype=tf.float32, size=seq_len / step_size, dynamic_size=False,
                                  infer_shape=True, clear_after_read=False)
        all_sub_features = TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                       infer_shape=True, clear_after_read=False)
        all_sub_goals = TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                    infer_shape=True, clear_after_read=False)

        # supervised pretraining for generator
        g_preds = TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)
        ta_emb_x = TensorArray(dtype=tf.float32, size=seq_len)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)
