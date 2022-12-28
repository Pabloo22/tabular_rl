import numpy as np
import tqdm
from tabular_rl.core import RLAgent, MarkovDecisionProcess


class DynamicProgramming(RLAgent):

    def __init__(self, env: MarkovDecisionProcess):
        super().__init__(env)
        self.state_value_array = None
        self.policy = None

    def select_action(self, state: int) -> int:
        pass

    def initialize(self) -> None:
        self.state_value_array = np.zeros(mdp.n_states)
        self.policy = np.zeros(mdp.n_states, dtype=int)

    def policy_evaluation(self, tol: float = 0.001, n_evaluations: int = 1000) -> bool:
        pass

    def policy_improvement(self) -> bool:
        pass

    # def fit(self, tol=0.001, max_evaluations=1, max_iters=100_000, use_tqdm=True) -> None:
    #
    #     for _ in tqdm.tqdm(range(100), disable=not use_tqdm):
    #
    #         for _ in range(max_evaluations):
    #             aux = np.fromfunction(np.vectorize(lambda state, action: sum(self.env.transition_probabilities_matrix[action,state,:]*(self.env.immediate_reward_matrix[action,state,:] + self.env.discount*self.V0))), (self.env.n_states, self.env.n_actions), dtype=int)
    #             pi = np.fromfunction(np.vectorize(lambda s, a: (a == self.policy[s])), (mdp.n_states, self.env.n_actions), dtype=int)
    #
    #             self.V0 = np.ma.masked_array(aux, mask=np.logical_not(pi)).compressed()  # V_k+1
    #         if np.array_equal(self.policy, aux.argmax(axis=1)):  # Si la politica no cambia, ha finalizado
    #             break
    #
    #         self.policy = aux.argmax(axis=1)

    def fit(self,
            tol: float = 0.001,
            max_policy_evaluations: int = 1,
            max_iters: int = 100_000,
            use_tqdm: bool = True) -> None:

        for _ in tqdm.tqdm(range(max_iters), disable=not use_tqdm):

            values_stable = self.policy_evaluation(tol, max_policy_evaluations)
            policy_stable = self.policy_improvement()

            if values_stable and policy_stable:
                break

    def save_learning(self, path: str) -> None:
        pass

    def load_learning(self, path: str) -> None:
        pass


if __name__ == '__main__':
    transition_probabilities = np.array([[[0.2, 0.5, 0.3],
                                          [0, 0.5, 0.5],
                                          [0, 0, 1]],

                                         [[0.3, 0.6, 0.1],
                                          [0.1, 0.6, 0.3],
                                          [0.05, 0.4, 0.55]]])

    reward_function = np.array([[[7, 6, 6],
                                 [0, 5, 1],
                                 [0, 0, -1]],

                                [[6, 6, -1],
                                 [7, 4, 0],
                                 [6, 3, -2]]])

    mdp = MarkovDecisionProcess(3, 2, transition_probabilities, reward_function, 1)
    agent = DynamicProgramming(mdp)
    agent.fit()
    print(agent.policy, agent.state_value_array)
