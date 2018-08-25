""" Trains an agent using stochastic policy gradients to play Pong. """
import gym
import numpy as np
import os
from rl.policy_gradient_pong.util import discount_rewards, load_model, log_episode, normalize, preprocess
from rl.policy_gradient_pong.util import save_model, sigmoid


# hyperparams
n_hidden = 200        # number of hidden layer neurons
batch_size = 10       # number episodes for param update
learning_rate = 1e-3
gamma = 0.99          # discount factor for reward
decay_rate = 0.99     # decay factor for RMSProp, leaky sum of grade^2
resume = False        # flag to resume from previous checkpoint
render = True
model_path = 'model.pkl'

# actions
UP = 2
DOWN = 3

env = gym.make('Pong-v0')


def choose_action(up_prob):
    """
    Note that it is standard practice to use a stochastic policy, meaning that
    we only produce a probability of going UP. Every iteration, we will sample
    from this distribution (i.e. toss a biased coin) to get the actual move.

    :param up_prob: (float) probability of going UP
    :return: (int) action
    """
    return UP if (np.random.uniform() < up_prob).all() else DOWN


def create_model(input_dim):
    return {
        'W1': np.random.randn(n_hidden, input_dim) / np.sqrt(input_dim),  # Xavier initialization
        'W2': np.random.randn(n_hidden) / np.sqrt(n_hidden)
    }


def policy_forward(model, x):
    """
    Intuitively, the neurons in the hidden layer (which have their weights arranged
    along the rows of W1) can detect various game scenarios (e.g. the ball is in the
    top, and our paddle is in the middle), and the weights in W2 can then decide if
    in each case we should be going UP or DOWN.

    :param model: (dict)
    :param x: (numpy float array) input, 80x80 difference frame
    :return: (float, numpy array) probability of going UP, and hidden state
    """
    h = np.dot(model['W1'], x)     # compute hidden layer neuron activations
    h[h < 0] = 0                   # ReLU non-linearity
    logp = np.dot(model['W2'], h)  # compute log probability of going UP
    p = sigmoid(logp)              # sigmoid "squashing" function to interval [0, 1]
    return p, h                    # return probability of going UP, and hidden state


def policy_backward(model, eph, epx, epdlogp):
    """
    backward pass

    :param model: (dict)
    :param eph: (numpy array) episode h, an array of intermediate hidden states
    :param epx: (numpy array) episode x
    :param epdlogp: (numpy array) episode log probabilities
    :return: (dict) gradients
    """
    dw2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backprop prelu
    dw1 = np.dot(dh.T, epx)
    return {'W1': dw1, 'W2': dw2}


def update_parameters(model, grad_buffer, rmsprop_cache):
    for k, v in model.items():
        grad = grad_buffer[k]  # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * grad ** 2
        model[k] += learning_rate * grad / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer


# noinspection SpellCheckingInspection
def init_model():
    """ initialize model """
    input_dim = 80 * 80   # input dimension: 80x80 grid
    if resume and os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model(input_dim)

    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}    # update buffers that add up gradients over a batch
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # RMSprop memory
    return input_dim, model, grad_buffer, rmsprop_cache


# noinspection SpellCheckingInspection
def train(input_dim, model, grad_buffer, rmsprop_cache):
    """ train model """
    # an image frame (a 210x160x3 byte array (integers from 0 to 255 giving pixel values)), 100,800 numbers total
    observation = env.reset()
    prev_x = None  # used in computing the difference frame
    xs, hs, dlogps, rewards = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_count = 0
    while True:
        if render:
            env.render()  # show the visual

        # preprocess the observation, convert 210x160x3 byte array to a 80x80 float vector
        cur_x = preprocess(observation)

        # set input to difference frame (i.e. subtraction of current and last frame) to detect motion
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        up_prob, h = policy_forward(model, x)   # probability of going UP
        action = choose_action(up_prob)  # roll the dice!

        # record various intermediates (needed for backprop)
        xs.append(x)  # observations
        hs.append(h)  # hidden states
        y = 1 if action == UP else 0  # a "fake label"

        # grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses)
        dlogps.append(y - up_prob)  # grads

        # step the environment and get new observation
        # a +1 reward if the ball went past the opponent, a -1 reward if we missed the ball, or 0 otherwise
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        rewards.append(reward)  # record reward (must be done after call to `step` to get reward for previous action)

        if done:  # an episode finished
            episode_count += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)          # episode x
            eph = np.vstack(hs)          # episode h
            epdlogp = np.vstack(dlogps)  # episode gradient
            epr = np.vstack(rewards)     # episode reward
            xs, hs, dlogps, rewards = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr, gamma)  # discounted episode reward

            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr = normalize(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens here!)
            grad = policy_backward(model, eph, epx, epdlogp)
            for k in model:
                grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform RMSprop parameter update every batch_size episodes
            if episode_count % batch_size == 0:
                update_parameters(model, grad_buffer, rmsprop_cache)

            # boring book keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            log_episode(reward_sum, running_reward)
            if episode_count % 100 == 0:
                save_model(model, model_path)

            reward_sum = 0
            observation = env.reset()
            prev_x = None

        # if reward != 0:
        #     print(('ep %d: game finished, reward: %f' % (episode_count, reward)) +
        #           ('' if reward == -1 else ' !!!!!!!!'))


train(*init_model())
