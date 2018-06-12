Navigate a Frozen Lake using a Markov Decision Process (MDP)
------------------------------------------------------------

The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable,
and others lead to the agent falling into the water. Additionally, the movement direction of the agent
is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a
walkable path to a goal tile.

A Markov Decision Process (MDP) is defined by how it changes states and how rewards are computed.

State transition is defined by :math:`P(s' |s,a)` - how likely you are to end at state :math:`s'`
if you take action :math:`a` from state :math:`s`. Now there's more than one way to define rewards,
but we'll use the :math:`r(s,a,s')` function for convenience.
