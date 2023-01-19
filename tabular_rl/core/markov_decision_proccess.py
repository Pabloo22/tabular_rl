from typing import Optional

import numpy as np
import warnings

from .tab_env import TabEnv


class MarkovDecisionProcess:
    """A Markov Decision Process (MDP).

    This class represents an MDP. Algorithms that require a model of the environment may use this class.

    If using this class directly, you must provide the transition matrix and immediate reward matrix. However, it is
    possible to create a new MDP by inheriting from this class and overriding the get_transition_probabilities and
    get_immediate_reward methods. That can be useful if the transition probabilities and immediate reward matrices
    are very large. That's the reason these matrices should not be accessed directly when implementing a new agent,
    but rather through the methods above.

    Args:
        n_states: Number of states.
        n_actions: Number of actions.
        discount: Discount factor. A float in [0, 1].
        transition_matrix: Transition probabilities. A numpy array of shape (n_actions, n_states, n_states).
        immediate_reward_matrix: Reward function. A numpy array of shape (n_actions, n_states, n_states).
        env: The environment of the MDP. This is useful to be able to play the game.

    Raises:
        ValueError: If the discount factor is not in [0, 1].
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 discount: float = 0.99,
                 transition_matrix: Optional[np.ndarray] = None,
                 immediate_reward_matrix: Optional[np.ndarray] = None,
                 env: Optional[TabEnv] = None):

        if discount < 0 or discount > 1:
            raise ValueError("The discount factor must be in the range [0, 1].")
        elif discount == 1:
            warnings.warn("The discount factor is 1. This can cause the DynamicProgramming agent to never "
                          "converge.")

        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount

        self.transition_matrix = transition_matrix
        self.immediate_reward_matrix = immediate_reward_matrix
        self.env = env

    def get_transition_probabilities(self, state: int, action: int) -> np.ndarray:
        """Returns an array with the transition probabilities for the given state and action for all possible
        next states."""
        return self.transition_matrix[action, state, :]

    def get_immediate_reward(self, state: int, action: int) -> np.ndarray:
        """Returns the expected immediate reward for the given state and action for all possible next states."""
        return self.immediate_reward_matrix[action, state, :]
