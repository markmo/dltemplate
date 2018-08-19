""" traffic simulators """
__credits__ = 'giorgio@ac.upc.edu'

import numpy as np
import os
import re
from rl.routing_optimization.util import softmax
from scipy.stats import norm


class OUProcess(object):
    """ Ornstein-Uhlenbeck Process """

    def __init__(self, processes, mu=0, theta=0.15, sigma=0.3):
        self.dt = 0.1
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.processes = processes
        self.state = np.ones(self.processes) * self.mu

    def reset(self):
        self.state = np.ones(self.processes) * self.mu

    def evolve(self):
        x = self.state
        dw = norm.rvs(scale=self.dt, size=self.processes)
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * dw
        self.state = x + dx
        return self.state


class Traffic(object):

    def __init__(self, n_nodes, traffic_type, capacity):
        self.n_nodes = n_nodes
        self.prev_traffic = None
        self.traffic_type = traffic_type
        self.capacity = capacity * n_nodes / (n_nodes - 1)
        self.traffic_gen_methods = {
            'norm': self.normal_traffic,
            'uni': self.uniform_traffic,
            'controlled': self.controlled_uniform_traffic,
            'exp': self.exp_traffic,
            'ou_process': self.ou_process_traffic,  # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
            'stat': self.stat_traffic,
            'stat_eq': self.stat_eq_traffic,
            'file': self.file_traffic,
            'dir': self.dir_traffic
        }
        self.static = None
        self.total_ou = OUProcess(1, self.capacity / 2, 0.1, self.capacity / 2)
        self.nodes_ou = OUProcess(self.n_nodes**2, 1, 0.1, 1)
        if self.traffic_type.startswith('dir:'):
            self.dir = sorted(os.listdir(self.traffic_type.split('dir:')[-1]), key=lambda x: natural_key(x))

    def normal_traffic(self):
        t = np.random.normal(self.capacity / 2, self.capacity / 2)
        return np.asarray(t * softmax(np.random.randn(self.n_nodes, self.n_nodes))).clip(min=0.001)

    def uniform_traffic(self):
        t = np.random.uniform(0, self.capacity * 1.25)
        return np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.n_nodes] * 2))).clip(min=0.001)

    def controlled_uniform_traffic(self):
        t = np.random.uniform(0, self.capacity * 1.25)
        if self.prev_traffic is None:
            self.prev_traffic = \
                np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.n_nodes] * 2))).clip(min=0.001)

        dist = [1]
        dist += [0] * (self.n_nodes**2 - 1)
        ch = np.random.choice(dist, [self.n_nodes] * 2)
        tt = np.multiply(self.prev_traffic, 1 - ch)
        nt = np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.n_nodes] * 2))).clip(min=0.001)
        nt = np.multiply(nt, ch)
        self.prev_traffic = tt + nt
        return self.prev_traffic

    def exp_traffic(self):
        a = np.random.exponential(size=self.n_nodes)
        b = np.random.exponential(size=self.n_nodes)
        t = np.outer(a, b)
        np.fill_diagonal(t, -1)
        t[t != -1] = np.asarray(np.random.exponential() * t[t != -1] / np.average(t[t != -1])).clip(min=0.001)
        return t

    def stat_traffic(self):
        if self.static is None:
            text = self.traffic_type.split('stat:')[-1]
            v = np.asarray(tuple(float(x) for x in text.split(',')[:self.n_nodes**2]))
            m = np.split(v, self.n_nodes)
            self.static = np.vstack(m)

        return self.static

    def stat_eq_traffic(self):
        if self.static is None:
            value = float(self.traffic_type.split('stat_eq:')[-1])
            self.static = np.full([self.n_nodes] * 2, value, dtype=float)

        return self.static

    def ou_process_traffic(self):
        t = self.total_ou.evolve()[0]
        nt = t * softmax(self.nodes_ou.evolve())
        i = np.split(nt, self.n_nodes)
        return np.vstack(i).clip(min=0.001)

    def file_traffic(self):
        if self.static is None:
            filename = 'traffic/' + self.traffic_type.split('file:')[-1]
            v = np.loadtxt(filename, delimiter=',')
            self.static = np.split(v, self.n_nodes)

        return self.static

    def dir_traffic(self):
        while len(self.dir) > 0:
            tm = self.dir.pop(0)
            if not tm.endswith('.txt'):
                continue

            filename = self.traffic_type.split('dir:')[-1] + '/' + tm
            v = np.loadtxt(filename, delimiter=',')
            return np.split(v, self.n_nodes)

        return False

    def generate(self):
        return self.traffic_gen_methods[self.traffic_type.split(':')[0]]()


def natural_key(text):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]
