import numpy as np
import tensorflow as tf


def discount_rewards(rewards, gamma=0.99):
    """ take 1D float array of rewards and compute discounted rewards """
    discounted = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted[t] = running_add

    return discounted


def reset_grad_buffer(grad_buffer):
    if type(grad_buffer) is not np.ndarray:
        grad_buffer = np.array(grad_buffer)

    grad_buffer *= 0
    return grad_buffer


def step_model(model, sess, xs, action):
    """
    This function uses our model to produce a new state
    when given a previous state and action.
    """
    pred = sess.run(model.pred_state, feed_dict={
        model.prev_state: np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
    })
    reward = pred[:, 4]
    obs = pred[:, 0:4]
    obs[:, 0] = np.clip(obs[:, 0], -2.4, 2.4)
    obs[:, 2] = np.clip(obs[:, 2], -0.4, 0.4)
    p_done = np.clip(pred[:, 5], 0, 1)
    done = (p_done > 0.1 or len(xs) >= 300)

    return obs, reward, done


def train(env, policy, model, model_batch_size, real_batch_size, max_episodes=5000):
    xs, ys, rs, ds = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_num = 1
    real_episodes = 1
    batch_size = real_batch_size

    draw_from_model = False  # When set to True, will use model for observations
    train_model = True  # Whether to train the model
    train_policy = False  # Whether to train the policy
    switch_point = 1

    p_state = None
    next_states_all = None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rendering = False
        obs = env.reset()
        # x = obs
        grad_buffer = sess.run(policy.tvars)
        grad_buffer = reset_grad_buffer(grad_buffer)

        while episode_num < max_episodes:
            # Start displaying environment once performance is acceptably high
            if rendering or (reward_sum / batch_size > 150 and not draw_from_model):
                env.render()
                rendering = True

            x = np.reshape(obs, [1, 4])
            prob = sess.run(policy.probability, feed_dict={policy.observations: x})
            action = 1 if np.random.uniform() < prob else 0

            xs.append(x)
            y = 1 if action == 0 else 0
            ys.append(y)

            if draw_from_model:
                obs, reward, done = step_model(model, sess, xs, action)
            else:
                obs, reward, done, _ = env.step(action)

            reward_sum += reward
            rs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
            ds.append(done * 1)

            if done:
                if not draw_from_model:
                    real_episodes += 1

                episode_num += 1

                # stack together all inputs, hidden states, action gradients,
                # and rewards for this episode
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(rs)
                epd = np.vstack(ds)
                xs, ys, rs, ds = [], [], [], []  # reset memory

                if train_model:
                    actions = np.array([np.abs(y - 1) for y in epy][:-1])
                    prev_states = epx[:-1, :]
                    prev_states = np.hstack([prev_states, actions])
                    next_states = epx[1:, :]
                    rewards = np.array(epr[1:, :])
                    dones = np.array(epd[1:, :])
                    next_states_all = np.hstack([next_states, rewards, dones])
                    feed_dict = {
                        model.prev_state: prev_states,
                        model.true_obs: next_states,
                        model.true_done: dones,
                        model.true_reward: rewards
                    }
                    loss, p_state, _ = sess.run([model.loss, model.pred_state, model.update_op], feed_dict)

                if train_policy:
                    discounted_epr = discount_rewards(epr).astype('float32')
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)
                    grads_t = sess.run(policy.new_grads, feed_dict={
                        policy.observations: epx,
                        policy.input_y: epy,
                        policy.advantage: discounted_epr
                    })

                    # If gradients become too large, end training process
                    if np.sum(grads_t[0] == grads_t[0]) == 0:
                        print('Gradients too large!')
                        break

                    for i, grad in enumerate(grads_t):
                        grad_buffer[i] += grad

                if switch_point + batch_size == episode_num:
                    switch_point = episode_num

                    if train_policy:
                        sess.run(policy.update_grads, feed_dict={
                            policy.w1_grad: grad_buffer[0],
                            policy.w2_grad: grad_buffer[1]
                        })
                        grad_buffer = reset_grad_buffer(grad_buffer)

                    if running_reward is None:
                        running_reward = reward_sum
                    else:
                        running_reward = running_reward * 0.99 + reward_sum * 0.01

                    if not draw_from_model:
                        print('World Perf: Episode %i  Reward %f  action: %i  mean reward %f' %
                              (real_episodes, reward_sum / real_batch_size, action,
                               running_reward / real_batch_size))

                        if reward_sum / batch_size > 200:
                            break

                    reward_sum = 0

                    # Once the model has been trained on 100 episodes,
                    # we start alternating between training the policy
                    # from the model and training the model from the
                    # real environment.
                    if episode_num > 100:
                        draw_from_model = not draw_from_model
                        train_model = not train_model
                        train_policy = not train_policy

                if draw_from_model:
                    obs = np.random.uniform(-0.1, 0.1, [4])  # Generate reasonable starting point
                    batch_size = model_batch_size
                else:
                    obs = env.reset()
                    batch_size = real_batch_size

    print('Num real episodes:', real_episodes)

    return p_state, next_states_all
