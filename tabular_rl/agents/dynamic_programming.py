import numpy as np
import tqdm

from tabular_rl.core import Agent, MarkovDecisionProcess


class DynamicProgramming(Agent):
    """A Dynamic Programming based agent.

    It makes use of a Markov Decision Process to perform policy evaluation and policy improvement.

    Example:
    >>> from tabular_rl.core import MarkovDecisionProcess
    >>> transition_probability_matrix = np.array([[[0.2, 0.5, 0.3], [0, 0.5, 0.5], [0, 0, 1]],
    ...                                            [[0.3, 0.6, 0.1],[0.1, 0.6, 0.3], [0.05, 0.4, 0.55]]])
    >>> reward_function = np.array([[[7, 6, 6], [0, 5, 1], [0, 0, -1]], [[6, 6, -1], [7, 4, 0], [6, 3, -2]]])
    >>> mdp = MarkovDecisionProcess(n_states=3, n_actions=2, discount=0.9,
    ...                             transition_matrix=transition_probability_matrix,
    ...                             immediate_reward_matrix=reward_function)
    >>> agent = DynamicProgramming(mdp)
    >>> agent.train()
    >>> print(agent.policy_, agent.state_value_array_)
    [1 1 1] [27.67244806 24.18025807 20.49361478]

    Args:
        mdp (MarkovDecisionProcess): The Markov Decision Process to solve. It must be an episodic MDP in order to be
        able to be played by the agent. However, it can be a non-episodic MDP if the agent is used for planning.
    """

    def __init__(self, mdp: MarkovDecisionProcess):
        super().__init__(mdp.env)
        self.mdp = mdp
        self.initialized = False

        # Attributes that will be initialized when the agent is trained
        self.state_value_array_ = None
        self.policy_ = None
        self._q_value_array_ = None

    def select_action(self, obs: any) -> int:
        """Selects an action according to the policy.

        Args:
            obs: The observation. An observation can be anything, but it is usually an integer or a tuple of integers.

        Returns:
            The action to perform as an integer in [0, n_actions).
        """
        state = self.env.obs2int(obs)
        return self.policy_[state]

    def _initialize(self) -> None:
        """Initializes the agent by setting the state value array and the policy to zero."""
        self.state_value_array_ = np.zeros(self.mdp.n_states)
        self.policy_ = np.zeros(self.mdp.n_states, dtype=int)

    def _get_q_value(self, state: int, action: int) -> np.ndarray:
        """Returns the q-value of a state-action pair."""
        # We need to convert `state` and `action` to integers to avoid errors when using `np.fromfunction`.
        state = int(state)
        action = int(action)
        transition_probabilities = self.mdp.get_transition_probabilities(state, action)
        immediate_rewards = self.mdp.get_immediate_reward(state, action)
        return np.sum(transition_probabilities * (immediate_rewards + self.mdp.discount * self.state_value_array_))

    def _policy_evaluation(self, tol: float = 0.001, n_evaluations: int = 1000) -> None:
        """Performs policy evaluation.

        Args:
            tol (float): The tolerance. If no change in the state value array is greater than this value then the
                policy evaluation is stopped.
            n_evaluations (int): The maximum number of policy evaluations to perform.
        """

        # True if the action has been selected by the policy
        mask = np.fromfunction(
            np.vectorize(lambda state, action: self.policy_[state] == action),
            (self.mdp.n_states, self.mdp.n_actions), dtype=int)

        for _ in range(n_evaluations):
            self._q_value_array_: np.ndarray = np.fromfunction(
                np.vectorize(self._get_q_value), (self.mdp.n_states, self.mdp.n_actions)
            )
            old_state_value_array = self.state_value_array_.copy()
            self.state_value_array_ = np.sum(self._q_value_array_ * mask, axis=1)
            # debug = self.env.transform_array(self.state_value_array_, transform_actions=False)

            if np.abs(self.state_value_array_ - old_state_value_array).max() < tol:
                break

    def train(self,
              tol: float = 0.001,
              max_policy_evaluations: int = 1,
              max_iters: int = 1_000,
              show_progress_bar: bool = True) -> None:
        """Trains the agent using the generalized policy iteration algorithm.

        Args:
            tol (float): The tolerance. This will be used as the stopping criterion for the
                policy evaluation and the policy iteration.
            max_policy_evaluations (int): The maximum number of policy evaluations to perform. If the policy evaluation
                is equal to one then we are performing value iteration.
            max_iters (int): The maximum number of iterations.
            show_progress_bar (bool): Whether to use tqdm to display the progress.
        """

        if not self.initialized:
            self._initialize()

        for _ in tqdm.trange(max_iters, disable=not show_progress_bar):

            old_state_value_array = self.state_value_array_.copy()
            self._policy_evaluation(tol, max_policy_evaluations)

            # Policy improvement
            self.policy_ = np.argmax(self._q_value_array_, axis=1)

            if np.abs(self.state_value_array_ - old_state_value_array).max() < tol:
                break

    def save_learning(self, path: str) -> None:
        """Saves the policy and state value array to a file."""
        np.savez(path, policy=self.policy_, state_value_array=self.state_value_array_)

    def load_learning(self, path: str) -> None:
        """Loads the policy and state value array from a file."""
        data = np.load(path)
        self.policy_ = data["policy"]
        self.state_value_array_ = data["state_value_array"]
        self.initialized = True


if __name__ == '__main__':
    # from tabular_rl.envs import CarRentalMDP, CarRentalEnv
    # from tabular_rl.agents import DoubleQLearning
    # from tabular_rl.core import MarkovDecisionProcess
    # import numpy as np

    # car_rental_env = CarRentalEnv(max_episode_length=10,
    #                               max_cars=5,
    #                               max_moves=3,
    #                               expected_rental_requests=(1, 2),
    #                               expected_rental_returns=(2, 1),
    #                               rental_credit=100,
    #                               move_cost=1,
    #                               discount=0.9,
    #                               seed=10)
    # car_rental_env = CarRentalEnv(max_episode_length=100)
    # car_rental_mdp = CarRentalMDP(car_rental_env)
    #
    # dp_agent = DynamicProgramming(car_rental_mdp)
    # dp_agent.train(tol=0.001, max_policy_evaluations=1, max_iters=1000)
    # car_rental_env.visualize_array(dp_agent.policy_, "DP Policy")
    # car_rental_env.visualize_array(dp_agent.state_value_array_, "State-Value Array", transform_actions=False)
    # print(car_rental_env.evaluate_agent(dp_agent, n_episodes=1_000))

    # q_learning_agent = DoubleQLearning(car_rental_env, init_method=100, epsilon=0.3, step_size=0.01)
    # q_learning_agent.train(n_episodes=100_000, eval_interval=10_000, use_tqdm=True)
    #
    # policy = np.zeros((car_rental_env.max_n_cars + 1, car_rental_env.max_n_cars + 1), dtype=int)
    # for i in range(car_rental_env.max_n_cars + 1):
    #     for j in range(car_rental_env.max_n_cars + 1):
    #         policy[car_rental_env.max_n_cars - i, j] = \
    #             car_rental_env.max_moves - q_learning_agent((i, j))
    #
    # car_rental_env.visualize_array(policy, "Q-Learning Policy")
    # print(car_rental_env.evaluate_agent(q_learning_agent, n_episodes=10_000))
    #
    # transition_probability_matrix = np.array([[[0.2, 0.5, 0.3], [0, 0.5, 0.5], [0, 0, 1]],
    #                                           [[0.3, 0.6, 0.1], [0.1, 0.6, 0.3], [0.05, 0.4, 0.55]]])
    #
    # reward_function = np.array([[[7, 6, 6], [0, 5, 1], [0, 0, -1]], [[6, 6, -1], [7, 4, 0], [6, 3, -2]]])

    # mdp = MarkovDecisionProcess(3, 2, 0.9, transition_probability_matrix, reward_function)
    # agent = DynamicProgramming(mdp)
    # agent.train(tol=1e-6, max_policy_evaluations=1, max_iters=1000)
    # print(agent.policy_, agent.state_value_array_, agent._q_value_array_, sep="\n")
    import doctest

    doctest.testmod()
