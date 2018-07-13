Bernoulli Bandits
-----------------

Implementation of several exploration strategies.

The bandit has K actions. An action produces a reward r of 1.0 with probability 0 <= θ_k <= 1,
which is unknown to the agent, but fixed over time.

The Agent's objective is to minimize regret over a fixed number T of action selections:

::

    p = T θ^* - sum{t=1 to T}(r_t)

where θ^* = max_k(θ_k)


Real-world analogy:

`Clinical trials <https://arxiv.org/pdf/1507.08025.pdf>`_ - we have K pills and T sick patients.
After taking a pill, the patient is cured with probability θ_k. Task is to find the most efficient pill.
