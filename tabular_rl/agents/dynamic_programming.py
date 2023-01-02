import numpy as np
import tqdm

from tabular_rl.core import RLAgent, MarkovDecisionProcess


class DynamicProgramming(RLAgent):
    """A Dynamic Programming based agent.

    It makes use of a Markov Decision Process to perform policy evaluation and policy improvement.

    Example:
    >>> transition_probability_matrix = np.array([[[0.2, 0.5, 0.3], [0, 0.5, 0.5], [0, 0, 1]
                                                  [[0.3, 0.6, 0.1],[0.1, 0.6, 0.3], [0.05, 0.4, 0.55]]])

    >>> reward_function = np.array([[[7, 6, 6], [0, 5, 1], [0, 0, -1]],
                                    [[6, 6, -1], [7, 4, 0], [6, 3, -2]]])

    >>> mdp = MarkovDecisionProcess(3, 2, 0.9, transition_probability_matrix, reward_function)
    >>> agent = DynamicProgramming(mdp)
    >>> agent.train()
    >>> print(agent.policy_, agent.state_value_array_)

    Args:
        mdp (MarkovDecisionProcess): The Markov Decision Process to solve. It must have an environment with a
            maximum number of episodes in order to be able to play the game.
        init_method (np.ndarray): The method to initialize the state value array. It can be "zeros", "uniform" or a
            float/int. If it is "zeros" then the state value array will be initialized to zeros. If it is "uniform"
            then the state value array will be initialized to a uniform distribution between 0 and 1. If it is a float
            or int then the state value array will be initialized to that value. The policy will be initialized to
            random values between 0 and the number of actions minus one independently of the init_method.
    """

    def __init__(self, mdp: MarkovDecisionProcess, init_method: np.ndarray = "zeros"):
        super().__init__(mdp.env)
        self.mdp = mdp
        self.init_method = init_method
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

        INIT_METHODS = {
            "zeros": np.zeros,
            "uniform": np.random.uniform,
        }

        if isinstance(self.init_method, (int, float)):
            self.state_value_array_ = np.full(self.mdp.n_states, self.init_method)
        elif self.init_method in INIT_METHODS:
            self.state_value_array_ = INIT_METHODS[self.init_method](self.mdp.n_states)
        else:
            raise ValueError(f"Invalid init_method: {self.init_method}")

        self.policy_ = np.random.randint(0, self.mdp.n_actions, self.mdp.n_states)

    def _get_q_value(self, state: int, action: int) -> np.ndarray:
        state = int(state)
        action = int(action)
        transition_probabilities = self.mdp.get_transition_probabilities(state, action)
        immediate_rewards = self.mdp.get_immediate_reward(state, action)
        return np.sum(transition_probabilities * (immediate_rewards + self.mdp.discount * self.state_value_array_))

    def _policy_evaluation(self, tol: float = 0.001, n_evaluations: int = 1000) -> None:

        mask = np.fromfunction(
            np.vectorize(lambda state, action: self.policy_[state] == action),
            (self.mdp.n_states, self.mdp.n_actions), dtype=int)

        for _ in range(n_evaluations):
            self._q_value_array_: np.ndarray = np.fromfunction(
                np.vectorize(self._get_q_value), (self.mdp.n_states, self.mdp.n_actions)
            )
            old_state_value_array = self.state_value_array_.copy()
            self.state_value_array_ = np.ma.masked_array(self._q_value_array_, mask=np.logical_not(mask)).compressed()

            if np.abs(self.state_value_array_ - old_state_value_array).max() < tol:
                break

    def train(self,
              tol: float = 0.001,
              max_policy_evaluations: int = 1,
              max_iters: int = 1_000,
              use_tqdm: bool = True) -> None:
        """Trains the agent.

        Args:
            tol (float): The tolerance for the policy evaluation. This will be used as the stopping criterion for the
                policy evaluation and the policy improvement.
            max_policy_evaluations (int): The maximum number of policy evaluations to perform. If the policy evaluation
                is equal to one then we are performing value iteration.
            max_iters (int): The maximum number of iterations.
            use_tqdm (bool): Whether to use tqdm to display the progress.
        """

        if not self.initialized:
            self._initialize()

        for _ in tqdm.tqdm(range(max_iters), disable=not use_tqdm):

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
    from tabular_rl.envs import CarRentalMDP, CarRentalEnv
    from tabular_rl.agents import DoubleQLearning

    car_rental_env = CarRentalEnv(max_episode_length=100)
    car_rental_mdp = CarRentalMDP(car_rental_env)

    dp_agent = DynamicProgramming(car_rental_mdp)
    dp_agent.train(tol=0.001, max_policy_evaluations=1, max_iters=1000)
    max_cars = car_rental_env.max_n_cars
    policy = np.zeros((max_cars + 1, max_cars + 1), dtype=int)
    for s, a in enumerate(dp_agent.policy_):
        n_cars_first_loc, n_cars_second_loc = np.unravel_index(s, policy.shape)
        policy[max_cars - n_cars_first_loc, n_cars_second_loc] = car_rental_env.max_moves - a
    print(car_rental_env.evaluate_agent(dp_agent, n_episodes=1000))
    print(policy)

    print("-" * 100)

    dql_agent = DoubleQLearning(car_rental_env)
    dql_agent.train(n_episodes=100_000, eval_interval=1000, n_episodes_eval=10)
    max_cars = car_rental_env.max_n_cars
    policy = np.zeros((max_cars + 1, max_cars + 1), dtype=int)
    for cars1 in range(max_cars + 1):
        for cars2 in range(max_cars + 1):
            policy[max_cars - cars1, cars2] = car_rental_env.max_moves - dql_agent.select_action((cars1, cars2))

    print(policy)
    print(car_rental_env.evaluate_agent(dql_agent, n_episodes=1000))
