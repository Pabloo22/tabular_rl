import numpy as np
from tqdm import tqdm
from tabular_rl.core import RLAgent, MarkovDecisionProcess


class DynamicProgramming(RLAgent):

    def __init__(self, mdp: MarkovDecisionProcess):
        super().__init__(mdp)
        self.V0 = np.zeros(mdp.n_states)  # DEBE CAMBIAR EN CADA ITERACION
        self.policy = np.zeros(mdp.n_states).astype(int)  # DEBE CAMBIAR EN CADA ITERACION

    def select_action(self, state: int) -> int:
        pass

    def fit(self) -> None:

        for _ in tqdm(range(100)):
            aux = np.fromfunction(np.vectorize(lambda state, action: sum(self.env.transition_probabilities_matrix[action,state,:]*(self.env.immediate_reward_matrix[action,state,:] + self.env.discount*self.V0))), (self.env.n_states, self.env.n_actions), dtype=int)
            pi = np.fromfunction(np.vectorize(lambda s, a: (a == self.policy[s])), (mdp.n_states, self.env.n_actions), dtype=int)

            self.V0 = np.ma.masked_array(aux, mask=np.logical_not(pi)).compressed()  # V_k+1
            if np.array_equal(self.policy, aux.argmax(axis=1)):  # Si la politica no cambia, ha finalizado
                break

            self.policy = aux.argmax(axis=1)

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
    print(agent.policy, agent.V0)
