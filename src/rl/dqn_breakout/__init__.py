import matplotlib.pyplot as plt
from rl.dqn_breakout.model_setup import DQNAgent, PreprocessAtariImage
from rl.dqn_breakout.util import make_env
from rl.util import evaluate, load_weights_into_target_network, play_and_record, ReplayBuffer
import tensorflow as tf


def run():
    env = make_env()
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n

    obs = env.reset()

    plt.imshow(obs[:, :, 0], interpolation='none', cmap='gray')
    plt.show()

    for _ in range(50):
        obs, _, _, _ = env.step(env.action_space.sample())

    plt.title('Game image')
    plt.imshow(env.render('rgb_array'))
    plt.show()
    plt.title('Agent observation (4 frames left to right)')
    plt.imshow(obs.transpose([0, 2, 1]).reshape([state_dim[0], -1]))
    plt.show()

    agent = DQNAgent('dqn_agent', state_dim, n_actions, epsilon=0.5)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    evaluate(env, agent, n_games=1)

    exp_replay = ReplayBuffer(10)

    for _ in range(30):
        exp_replay.add(env.reset(), env.action_space.sample(), 1.0, env.reset(), done=False)

    obsvs_batch, actions_batch, rewards_batch, next_obsvs_batch, is_done_batch = exp_replay.sample(5)

    assert len(exp_replay) == 10, 'experience replay size should be 10 - maximum capacity'

    exp_replay = ReplayBuffer(20000)
    play_and_record(agent, env, exp_replay, n_steps=10000)

    assert len(exp_replay) == 10000, 'play_and_record should have added exactly 10000 steps, ' \
                                     'but instead added %i' % len(exp_replay)

    is_done = list(zip(*exp_replay.storage))[-1]

    target_network = DQNAgent('target_network', state_dim, n_actions)
    load_weights_into_target_network(agent, target_network)

    # placeholders that will be fed with exp_replay.sample(batch_size)
    obsvs_ph = tf.placeholder(tf.float32, shape=[None,] + state_dim)
    actions_ph = tf.placeholder(tf.int32, shape=[None])
    rewards_ph = tf.placeholder(tf.float32, shape=[None])
    next_obsvs_ph = tf.placeholder(tf.float32, shape=[None,] + state_dim)
    is_done_ph = tf.placeholder(tf.float32, shape=[None])

    is_not_done = 1 - is_done_ph
    gamma = 0.99

    current_q_values = agent.get_symbolic_q_values(obsvs_ph)
    current_action_q_values = tf.reduce_sum(tf.one_hot(actions_ph, n_actions) * current_q_values, axis=1)
    next_q_values_target = target_network.get_symbolic_q_values(next_obsvs_ph)
    next_state_values_target = tf.reduce_max(next_q_values_target, axis=-1)
    reference_q_values = rewards_ph + gamma * next_state_values_target * is_not_done
    td_loss = (current_action_q_values - reference_q_values)**2
    td_loss = tf.reduce_mean(td_loss)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(td_loss, var_list=agent.weights)






