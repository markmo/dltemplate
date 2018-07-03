import numpy as np
from rl.aac_kungfumaster.model_setup import ActorCritic, Agent
from rl.aac_kungfumaster.util import EnvBatch, make_env, sample_actions
import tensorflow as tf


def test_actor_critic_model():
    """ Specific to default setup for Kung Fu Master """
    env = make_env()
    env_batch = EnvBatch(10)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = Agent('agent', obs_shape, n_actions)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_states = env_batch.reset()
    batch_actions = sample_actions(agent.step(sess, batch_states))
    batch_next_states, batch_rewards, batch_done, _ = env_batch.step(batch_actions)

    model = ActorCritic(obs_shape, n_actions, agent)

    actor_loss, critic_loss, advantage, entropy = \
        sess.run([model.actor_loss, model.critic_loss, model.advantage, model.entropy], feed_dict={
            model.states_ph: batch_states,
            model.actions_ph: batch_actions,
            model.next_states_ph: batch_states,
            model.rewards_ph: batch_rewards,
            model.is_done_ph: batch_done
        })

    sess.close()

    assert abs(actor_loss) < 100 and abs(critic_loss) < 100, 'losses seem abnormally large'
    assert 0 <= entropy.mean() <= np.log(n_actions), 'impossible entropy value, check the formula'
    assert np.log(n_actions) / 2 < entropy.mean(), 'Entropy is too low for untrained agent'
