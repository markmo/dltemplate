Vanilla Policy Gradient Agent
-----------------------------

This tutorial contains a simple example of how to build a policy-gradient based
agent that can solve the CartPole problem.

The agent is capable of taking in an observation of the world, then taking actions
which provide the optimal reward not just in the present, but over the long run.

To take reward over time into account, we need to update our agent with more than
one experience at a time. To accomplish this, we collect experiences in a buffer,
and then occasionally use them to update the agent all at once. These sequences of
experience are sometimes referred to as rollouts, or experience traces.

Rewards are discounted over time. We use this modified reward as an estimation of
the advantage in our loss equation.
