import numpy as np

from typing import Optional
from .tab_env import TabEnv


class MarkovDecisionProcess:
    """A Markov Decision Process (MDP).

    This class is used to represent an MDP. It is used by algorithms which require a model of the environment.
    An example of such an algorithm is policy iteration by dynamic programming.

    If using this class directly, the transition matrix and immediate reward matrix must be provided. However, it is
    possible to create a new MDP by inheriting from this class and overriding the get_transition_probabilities and
    get_immediate_reward methods. This is useful if the transition probabilities and immediate rewards are very large.
    That's the reason these matrices should not be accessed directly when implementing a new agent, but rather through
    the methods above.

    Args:
        n_states: Number of states.
        n_actions: Number of actions.
        discount: Discount factor. A float in [0, 1).
        transition_matrix: Transition probabilities. A numpy array of shape (n_actions, n_states, n_states).
        immediate_reward_matrix: Reward function. A numpy array of shape (n_actions, n_states, n_states).
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 discount: float = 0.99,
                 transition_matrix: Optional[np.ndarray] = None,
                 immediate_reward_matrix: Optional[np.ndarray] = None,
                 env: Optional[TabEnv] = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount

        self._transition_matrix = transition_matrix
        self._immediate_reward_matrix = immediate_reward_matrix
        self.env = env

    def get_transition_probabilities(self, state: int, action: int) -> np.ndarray:
        """Returns the transition probabilities for the given state and action."""
        return self._transition_matrix[action, state, :]

    def get_immediate_reward(self, state: int, action: int) -> np.ndarray:
        """Returns the expected immediate reward for the given state and action for all possible next states."""
        return self._immediate_reward_matrix[action, state, :]
