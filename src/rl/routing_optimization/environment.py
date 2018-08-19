""" environments """
__credits__ = 'giorgio@ac.upc.edu'

import networkx as nx
import numpy as np
from rl.routing_optimization.traffic import Traffic
from rl.routing_optimization.util import format_float, ROOT, softmax
from scipy import stats
import subprocess

OM_TRAFFIC = 'traffic.txt'
OM_BALANCING = 'balancing.txt'
OM_ROUTING = 'routing.txt'
OM_DELAY = 'delay.txt'
TRAFFIC_LOG = 'traffic_log.csv'
BALANCING_LOG = 'balancing_log.csv'
REWARD_LOG = 'reward_log.csv'
WHOLE_LOG = 'log.csv'
OM_LOG = 'omnet_log.csv'


def matrix_to_rl(matrix):
    return matrix[(matrix != -1)]


matrix_to_log_v = matrix_to_rl


def matrix_to_omnet_v(matrix):
    return matrix.flatten()


def vector_to_file(vector, filename, mode):
    with open(filename, mode) as f:
        return f.write(','.join(format_float(x) for x in vector) + '\n')


def file_to_csv(filename):
    with open(filename, 'r') as f:
        return f.readline().strip().strip(',')


def csv_to_matrix(row, n_nodes):
    v = np.asarray(tuple(float(x) for x in row.split(',')[:n_nodes**2]))
    m = np.split(v, n_nodes)
    return np.vstack(m)


def csv_to_lost(row):
    return float(row.split(',')[-1])


def rl_to_matrix(vector, n_nodes):
    m = np.split(vector, n_nodes)
    for i in range(n_nodes):
        m[i] = np.insert(m[i], i, -1)

    return np.vstack(m)


def rl_state(env):
    if env.state_type == 'rt':
        return np.concatenate((matrix_to_rl(env.env_b), matrix_to_rl(env.env_t)))

    if env.state_type == 't':
        return matrix_to_rl(env.env_t)

    raise ValueError("Value of env.state_type expected to be one of ['rt', 't']")


def rl_reward(env):
    delay = np.asarray(env.env_d)
    mask = delay == np.inf
    delay[mask] = len(delay) * np.max(delay[~mask])
    if env.reward_fn == 'avg':
        reward = -np.mean(matrix_to_rl(delay))
    elif env.reward_fn == 'max':
        reward = -np.max(matrix_to_rl(delay))
    elif env.reward_fn == 'axm':
        reward = -(np.mean(matrix_to_rl(delay)) + np.max(matrix_to_rl(delay))) / 2
    elif env.reward_fn == 'geo':
        reward = -stats.gmean(matrix_to_rl(delay))
    elif env.reward_fn == 'lost':
        reward = -env.env_l
    else:
        raise ValueError("Value of env.reward_fn expected to be one of ['avg', 'max', 'axm', 'geo', 'lost']")

    return reward


# noinspection SpellCheckingInspection
def omnet_wrapper(env):
    if env.env == 'label':
        sim = 'router'
    elif env.env == 'balancing':
        sim = 'balancer'
    else:
        raise ValueError("Value of env.env expected to be one of ['label', 'balancing']")

    prefix = ROOT
    # if env.cluster == 'arvei':
    #     prefix = '/scratch/nas/1/giorgio/rlnet/'

    sim_exe = '{}omnet/{}/networkRL'.format(prefix, sim)
    sim_folder = '{}omnet/{}/'.format(prefix, sim)
    sim_ini = '{}omnet/{}/omnetpp.ini'.format(prefix, sim)

    try:
        omnet_output = subprocess.check_output([sim_exe, '-n', sim_folder, sim_ini, env.folder + 'folder.ini']).decode()
    except Exception as e:
        # noinspection PyUnresolvedReferences
        omnet_output = e.stdout.decode()

    if 'Error' in omnet_output:
        omnet_output = omnet_output.replace(',', '')
        o_u_l = [line.strip() for line in omnet_output.split('\n') if line is not '']
        omnet_output = ','.join(o_u_l[4:])
    else:
        omnet_output = 'OK'

    vector_to_file([omnet_output], env.folder + OM_LOG, 'a')


def ned_to_capacity(env):
    if env.env == 'label':
        sim = 'router'
    elif env.env == 'balancing':
        sim = 'balancer'
    else:
        raise ValueError("Value of env.env expected to be one of ['label', 'balancing']")

    ned_file = '{}omnet/{}/NetworkAll.ned'.format(ROOT, sim)
    capacity = 0
    with open(ned_file) as f:
        for line in f:
            if 'SlowChannel' in line and '<-->' in line:
                capacity += 3
            elif 'MediumChannel' in line and '<-->' in line:
                capacity += 5
            elif 'FastChannel' in line and '<-->' in line:
                capacity += 10
            elif 'Channel' in line and '<-->' in line:
                capacity += 10

    return capacity or None


class OmnetBalancerEnv(object):

    def __init__(self, constants, folder):
        self.env = 'balancing'
        self.routing = 'balancer'
        self.folder = folder
        self.active_nodes = constants['active_nodes']
        self.action_type = constants['action_type']
        self.a_dim = self.active_nodes**2 - self.active_nodes  # routing table minus diagonal
        self.s_dim = self.active_nodes**2 - self.active_nodes  # traffic minus diagonal
        self.state_type = constants['state_type']
        if self.state_type == 'rt':
            self.s_dim *= 2  # traffic + routing table minus diagonals

        if 'max_delta' in constants:
            self.max_delta = constants['max_delta']

        self.reward_fn = constants['reward_fn']
        capacity = self.active_nodes * (self.active_nodes - 1)
        self.traffic = constants['traffic']
        self.t_gen = Traffic(self.active_nodes, self.traffic, capacity)
        self.cluster = constants['cluster'] if 'cluster' in constants else False
        self.env_t = np.full([self.active_nodes] * 2, -1.0, dtype=float)  # traffic
        self.env_b = np.full([self.active_nodes] * 2, -1.0, dtype=float)  # balancing
        self.env_d = np.full([self.active_nodes] * 2, -1.0, dtype=float)  # delay
        self.env_l = -1.0  # lost packets
        self.counter = 0

    def update_env_t(self, matrix):
        self.env_t = np.asarray(matrix)
        np.fill_diagonal(self.env_t, -1)

    def update_env_b(self, matrix):
        self.env_b = np.asarray(matrix)
        np.fill_diagonal(self.env_b, -1)

    def update_env_d(self, matrix):
        self.env_d = np.asarray(matrix)
        np.fill_diagonal(self.env_d, -1)

    def update_env_l(self, scalar):
        self.env_l = scalar

    # noinspection PyTypeChecker
    def log_header(self):
        nice_matrix = np.chararray([self.active_nodes] * 2, itemsize=20)
        for i in range(self.active_nodes):
            for j in range(self.active_nodes):
                nice_matrix[i][j] = str(i) + '-' + str(j)

        np.fill_diagonal(nice_matrix, '_')
        nice_list = list(nice_matrix[(nice_matrix != b'_')])
        th = ['t' + x.decode('ascii') for x in nice_list]
        rh = ['r' + x.decode('ascii') for x in nice_list]
        dh = ['d' + x.decode('ascii') for x in nice_list]
        if self.state_type == 't':
            sh = ['s' + x.decode('ascii') for x in nice_list]
        elif self.state_type == 'rt':
            sh = ['sr' + x.decode('ascii') for x in nice_list] + ['st' + x.decode('ascii') for x in nice_list]
        else:
            raise ValueError("Value of self.state_type expected to be one of ['rt', 't']")

        ah = ['a' + x.decode('ascii') for x in nice_list]
        header = ['counter'] + th + rh + dh + ['lost'] + sh + ah + ['reward']
        vector_to_file(header, self.folder + WHOLE_LOG, 'w')

    # noinspection PyMethodMayBeStatic
    def render(self):
        return True

    def reset(self):
        if self.counter != 0:
            return None

        self.log_header()

        # balancing
        self.update_env_b(np.full([self.active_nodes] * 2, 0.5, dtype=float))
        if self.action_type == 'delta':
            vector_to_file(matrix_to_omnet_v(self.env_b), self.folder + OM_BALANCING, 'w')

        # traffic
        self.update_env_t(self.t_gen.generate())
        vector_to_file(matrix_to_omnet_v(self.env_t), self.folder + OM_TRAFFIC, 'w')
        return rl_state(self)

    def step(self, action):
        self.counter += 1

        # define action: 'new' or 'delta'
        if self.action_type == 'new':
            # bind the action
            self.update_env_b(rl_to_matrix(np.clip(action, 0, 1), self.active_nodes))
        elif self.action_type == 'delta':
            # bind the action
            self.update_env_b(rl_to_matrix(np.clip(action * self.max_delta + matrix_to_rl(self.env_b), 0, 1),
                                           self.active_nodes))

        # write to file input to OMNeT: Balancing
        vector_to_file(matrix_to_omnet_v(self.env_b), self.folder + OM_BALANCING, 'w')

        # execute OMNeT
        omnet_wrapper(self)

        # read OMNeT's output: Delay and Lost packets
        om_output = file_to_csv(self.folder + OM_DELAY)
        self.update_env_d(csv_to_matrix(om_output, self.active_nodes))
        self.update_env_l(csv_to_lost(om_output))

        reward = rl_reward(self)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARD_LOG, 'a')
        s = rl_state(self)
        log = np.concatenate(([self.counter],
                              matrix_to_log_v(self.env_t),
                              matrix_to_log_v(self.env_b),
                              matrix_to_log_v(self.env_d),
                              [self.env_l], s, action, [-reward]))
        vector_to_file(log, self.folder + WHOLE_LOG, 'a')

        # generate traffic for next iteration
        self.update_env_t(self.t_gen.generate())

        # write to file input for OMNeT: Traffic or do nothing if static
        if self.traffic.split(':')[0] not in ('stat', 'stat_eq', 'file'):
            vector_to_file(matrix_to_omnet_v(self.env_t), self.folder + OM_TRAFFIC, 'w')

        new_state = rl_state(self)

        return new_state, reward, 0

    def end(self):
        return


class OmnetLinkweightEnv(object):
    """ Label environment """

    def __init__(self, constants, folder):
        self.env = 'label'
        self.routing = 'Linkweight'
        self.folder = folder
        self.active_nodes = constants['active_nodes']
        self.action_type = constants['action_type']
        topology = ROOT + 'omnet/router/NetworkAll.matrix'
        self.graph = nx.Graph(np.loadtxt(topology, dtype=int))
        if self.active_nodes != self.graph.number_of_nodes():
            raise ValueError("hyperparam 'active_nodes' does not equal number of nodes in graph")

        ports = ROOT + 'omnet/router/NetworkAll.ports'
        self.ports = np.loadtxt(ports, dtype=int)
        self.a_dim = self.graph.number_of_edges()
        self.s_dim = self.active_nodes**2 - self.active_nodes  # traffic minus diagonal
        self.state_type = constants['state_type']
        if self.state_type == 'rt':
            self.s_dim *= 2  # traffic + routing table minus diagonals

        self.reward_fn = constants['reward_fn']
        capacity = self.active_nodes * (self.active_nodes - 1)
        self.traffic = constants['traffic']
        self.t_gen = Traffic(self.active_nodes, self.traffic, capacity)
        self.cluster = constants['cluster'] if 'cluster' in constants else None
        self.env_t = np.full([self.active_nodes] * 2, -1.0, dtype=float)  # traffic
        self.env_w = np.full([self.a_dim], -1.0, dtype=float)             # weights
        self.env_r = np.full([self.active_nodes] * 2, -1.0, dtype=int)    # routing
        self.env_rn = np.full([self.active_nodes] * 2, -1.0, dtype=int)   # routing (nodes)
        self.env_d = np.full([self.active_nodes] * 2, -1.0, dtype=float)  # delay
        self.env_l = -1.0  # lost packets
        self.counter = 0

    def update_env_t(self, matrix):
        self.env_t = np.asarray(matrix)
        np.fill_diagonal(self.env_t, -1)

    def update_env_w(self, vector):
        self.env_w = np.asarray(softmax(vector))

    def update_env_r(self):
        weights = {}
        for e, w in zip(self.graph.edges(), self.env_w):
            weights[e] = w

        nx.set_edge_attributes(self.graph, name='weight', values=weights)
        routing_nodes = np.full([self.active_nodes] * 2, -1.0, dtype=int)
        routing_ports = np.full([self.active_nodes] * 2, -1.0, dtype=int)
        all_shortest = nx.all_pairs_dijkstra_path(self.graph)
        for s in range(self.active_nodes):
            for d in range(self.active_nodes):
                if s != d:
                    # TODO
                    # must be updated for networkx 2.x
                    # `nx.all_pairs_dijkstra_path` returns a generator instead of a list
                    next_ = all_shortest[s][d][1]
                    port = self.ports[s][next_]
                    routing_nodes[s][d] = next_
                    routing_ports[s][d] = port
                else:
                    routing_nodes[s][d] = -1
                    routing_ports[s][d] = -1

        self.env_r = np.asarray(routing_ports)
        self.env_rn = np.asarray(routing_nodes)

    def update_env_r_from_r(self, routing):
        routing_nodes = np.fromstring(routing, sep=',', dtype=int)
        m = np.split(np.asarray(routing_nodes), self.active_nodes)
        routing_nodes = np.vstack(m)
        routing_ports = np.zeros([self.active_nodes] * 2, dtype=int)
        for s in range(self.active_nodes):
            for d in range(self.active_nodes):
                if s != d:
                    next_ = routing_nodes[s][d]
                    port = self.ports[s][next_]
                    routing_ports[s][d] = port
                else:
                    routing_ports[s][d] = -1

        self.env_r = np.asarray(routing_ports)
        self.env_rn = np.asarray(routing_nodes)

    def update_env_d(self, matrix):
        self.env_d = np.asarray(matrix)
        np.fill_diagonal(self.env_d, -1)

    def update_env_l(self, scalar):
        self.env_l = scalar

    # noinspection PyTypeChecker
    def log_header(self, easy=False):
        nice_matrix = np.chararray([self.active_nodes] * 2, itemsize=20)
        for i in range(self.active_nodes):
            for j in range(self.active_nodes):
                nice_matrix[i][j] = str(i) + '-' + str(j)

        np.fill_diagonal(nice_matrix, '_')
        nice_list = list(nice_matrix[(nice_matrix != b'_')])
        th = ['t' + x.decode('ascii') for x in nice_list]
        rh = ['r' + x.decode('ascii') for x in nice_list]
        dh = ['d' + x.decode('ascii') for x in nice_list]
        ah = ['a' + str(x[0]) + '-' + str(x[1]) for x in self.graph.edges()]
        header = ['counter'] + th + rh + dh + ['lost'] + ah + ['reward']
        if easy:
            header = ['counter', 'lost', 'AVG', 'MAX', 'AXM', 'GEO']

        vector_to_file(header, self.folder + WHOLE_LOG, 'w')

    def render(self):
        return

    # noinspection SpellCheckingInspection
    def reset(self, easy=False):
        if self.counter != 0:
            return None

        self.log_header(easy)

        # routing
        self.update_env_w(np.full([self.a_dim], 0.5, dtype=float))
        self.update_env_r()
        if self.action_type == 'delta':
            vector_to_file(matrix_to_omnet_v(self.env_r), self.folder + OM_ROUTING, 'w')
            # verify file position and format (separator, matrix/vector) np.savetxt('tmp.txt', routing, fmt='%d')

        # traffic
        self.update_env_t(self.t_gen.generate())

        vector_to_file(matrix_to_omnet_v(self.env_t), self.folder + OM_TRAFFIC, 'w')

        return rl_state(self)

    # noinspection SpellCheckingInspection
    def step(self, action):
        self.counter += 1
        self.update_env_w(action)
        self.update_env_r()

        # write to file input for OMNeT: Routing
        vector_to_file(matrix_to_omnet_v(self.env_r), self.folder + OM_ROUTING, 'w')
        # verify file position and format (separator, matrix/vector) np.savetxt('tmp.txt', routing, fmt='%d')

        # execute OMNet
        omnet_wrapper(self)

        # read OMNeT's output: Delay and Lost packets
        om_output = file_to_csv(self.folder + OM_DELAY)
        self.update_env_d(csv_to_matrix(om_output, self.active_nodes))
        self.update_env_l(csv_to_lost(om_output))

        reward = rl_reward(self)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARD_LOG, 'a')
        # s = rl_state(self)
        log = np.concatenate(([self.counter],
                              matrix_to_log_v(self.env_t),
                              matrix_to_log_v(self.env_rn),
                              matrix_to_log_v(self.env_d),
                              [self.env_l],
                              matrix_to_log_v(self.env_w),
                              [-reward]))
        vector_to_file(log, self.folder + WHOLE_LOG, 'a')

        # generate traffic for next iteration
        self.update_env_t(self.t_gen.generate())

        # write to file input for OMNeT: Traffic or do nothing if static
        if self.traffic.split(':')[0] not in ('stat', 'stat_eq', 'file'):
            vector_to_file(matrix_to_omnet_v(self.env_t), self.folder + OM_TRAFFIC, 'w')

        new_state = rl_state(self)

        return new_state, reward, 0

    # noinspection SpellCheckingInspection
    def easy_step(self, action):
        self.counter += 1
        self.update_env_r_from_r(action)

        # write to file input for OMNeT: Routing
        vector_to_file(matrix_to_omnet_v(self.env_r), self.folder + OM_ROUTING, 'w')
        # verify file position and format (separator, matrix/vector) np.savetxt('tmp.txt', routing, fmt='%d')

        # execute OMNeT
        omnet_wrapper(self)

        # read OMNeT's output: Delay and Lost packets
        om_output = file_to_csv(self.folder + OM_DELAY)
        self.update_env_d(csv_to_matrix(om_output, self.active_nodes))
        self.update_env_l(csv_to_lost(om_output))

        reward = rl_reward(self)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARD_LOG, 'a')
        # s = rl_state(self)
        log = np.concatenate(([self.counter],
                              [self.env_l],
                              [np.mean(matrix_to_rl(self.env_d))],
                              [np.max(matrix_to_rl(self.env_d))],
                              [(np.mean(matrix_to_rl(self.env_d)) + np.max(matrix_to_rl(self.env_d))) / 2],
                              [stats.gmean(matrix_to_rl(self.env_d))]))
        vector_to_file(log, self.folder + WHOLE_LOG, 'a')

        # generate traffic for next iteration
        self.update_env_t(self.t_gen.generate())

        # write to file input for OMNeT: Traffic or do nothing if static
        if self.traffic.split(':')[0] not in ('stat', 'stat_eq', 'file', 'dir'):
            vector_to_file(matrix_to_omnet_v(self.env_t), self.folder + OM_TRAFFIC, 'w')

        new_state = rl_state(self)

        return new_state, reward, 0

    def end(self):
        return
