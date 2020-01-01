import random
import numpy as np
from A3C.Config import *


class Agent:
    frames = 0

    def __init__(self, brain, use_epsilon=True):
        self.brain = brain
        if use_epsilon:
            self.eps_start = EPS_START
            self.eps_end = EPS_STOP
        else:
            self.eps_start = 0
            self.eps_end = 0
        self.eps_steps = EPS_STEPS
        self.num_actions = brain.num_actions

        self.memory = []  # used for n_step return
        self.R = 0.

    def _get_epsilon(self):
        Agent.frames = Agent.frames + 1
        if Agent.frames >= self.eps_steps:
            return self.eps_end
        else:
            return self.eps_start + Agent.frames * (self.eps_end - self.eps_start) / self.eps_steps

    def act(self, s):
        if random.random() < self._get_epsilon():
            return random.randint(0, self.num_actions - 1)
        else:
            s = np.array([s])
            p = self.brain.predict_p(s)[0]
            a = np.random.choice(self.num_actions, p=p)
            return a

    def train(self, s, a, r, s_):
        a_cats = np.zeros(self.num_actions)  # turn action into one-hot representation
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))
        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                self._push_sample()
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            self.R = 0

        if len(self.memory) == N_STEP_RETURN:
            self._push_sample()
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect

    def _push_sample(self):
        s = self.memory[0][0]
        a = self.memory[0][1]
        r = self.R
        s_ = self.memory[-1][3]
        self.brain.train_push(s, a, r, s_)
