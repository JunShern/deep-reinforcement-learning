import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.actions = list(range(nA))
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.001
        self.gamma = 1
        print("alpha", self.alpha, "e_decay", self.epsilon_decay, "e_min", self.epsilon_min, "gamma", self.gamma)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        # Follow epsilon-greedy policy
        greedy_choice = np.argmax(self.Q[state])
        random_choice = np.random.choice(self.actions)
        epsilon_greedy_choice = np.random.choice(
            [greedy_choice, random_choice], 
            p = [1-self.epsilon, self.epsilon]
        )
        return epsilon_greedy_choice

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Calculate expected return
        next_G = 0
        if not done:
            next_G = self.epsilon * sum([self.Q[next_state][action] for action in self.actions]) / self.nA + (1 - self.epsilon) * max(self.Q[next_state])
        
        # Update Q
        self.Q[state][action] += self.alpha * ((reward + self.gamma * next_G) - self.Q[state][action])