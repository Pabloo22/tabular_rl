import numpy as np
from dataclasses import dataclass


@dataclass(slots=True)
class MarkovDecisionProcess:
    """A Markov Decision Process (MDP).

    Arguments:
        n_states: Number of states.
        n_actions: Number of actions.
        transition_probabilities_matrix: Transition probabilities. A numpy array of shape (n_states, n_actions, n_states).
        immediate_reward_matrix: Reward function. A numpy array of shape (n_states, n_actions, n_states).
        discount: Discount factor. A float in [0, 1).
    """

    n_states: int
    n_actions: int
    transition_probabilities_matrix: np.ndarray
    immediate_reward_matrix: np.ndarray
    discount: float


if __name__ == "__main__":
    mdp = MarkovDecisionProcess(1, 1, np.array([1]), np.array([1]), 1)
    print(mdp)
