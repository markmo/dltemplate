

def get_burst_prob_actions(burst_prob_params):
    pass


def get_simplified_reward(action, reward_params, print_mode, burst_prob_user_selector,
                          original_throughput, hard_throughput_limit_flag, new_cell_throughput=None,
                          kb_mb_converter=1):
    """
    Map B to B'
    new_cell_throughput = original_throughput * (1 - action)
    B' = new_cell_throughput (KB/sec)
    A = action = burst_prob [0..1] (no unit)
    T = control_interval = 60 sec
    data_mb = B' * A * T

    :param action:
    :param reward_params:
    :param print_mode:
    :param burst_prob_user_selector:
    :param original_throughput:
    :param hard_throughput_limit_flag:
    :param new_cell_throughput:
    :param kb_mb_converter:
    :return:
    """
    burst_prob = action[0]
    control_interval = reward_params['control_interval']
    ppc_scheduled_data_mb = new_cell_throughput * burst_prob * control_interval / kb_mb_converter

    # if PPC user and regular user have same burst probabilities?
    if burst_prob_user_selector == 'same_as_ppc':
        user_burst_prob = burst_prob
    else:
        user_burst_prob = reward_params['user_burst_prob_mean']

    user_orig_data_mb = original_throughput * user_burst_prob * control_interval / kb_mb_converter
    user_new_data_mb = new_cell_throughput * user_burst_prob * control_interval / kb_mb_converter
    user_lost_data_mb = user_orig_data_mb - user_new_data_mb

    if hard_throughput_limit_flag:
        # apply a penalty if below the throughput limit
        hard_throughput_limit = reward_params['hard_throughput_limit']
        if new_cell_throughput < hard_throughput_limit:
            hard_throughput_limit_mb = burst_prob * (hard_throughput_limit - new_cell_throughput) * \
                                       control_interval / kb_mb_converter
        else:
            hard_throughput_limit_mb = 0.
    else:
        hard_throughput_limit_mb = 0.

    alpha, beta, kappa = [reward_params[k] for k in ('alpha', 'beta', 'kappa')]
    reward = alpha * ppc_scheduled_data_mb - beta * user_lost_data_mb - kappa * hard_throughput_limit_mb

    if print_mode:
        print('original_throughput:', original_throughput, 'new_cell_throughput:', new_cell_throughput)
        print('ppc_burst_prob:', burst_prob, 'user_burst_prob:', user_burst_prob)
        print('ppc_scheduled_data_mb:', ppc_scheduled_data_mb, 'user_lost_data_mb:', user_lost_data_mb)
        print('hard_throughput_limit_mb:', hard_throughput_limit_mb, 'reward:', reward)

    return reward, action, ppc_scheduled_data_mb, user_lost_data_mb, hard_throughput_limit_mb


def report_rewards(state, burst_prob, reward, reward_history, iteration_index,
                   ppc_data_mb_scheduled, user_lost_data_mb, print_mode,
                   original_throughput, new_throughput, throughput_var, batch_number):
    pass
