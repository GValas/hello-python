import threading
import time
import gym
from A3C.Config import *


class Environment(threading.Thread):
    @staticmethod
    def get_states_actions():
        env = gym.make(ENV)
        s = env.observation_space.shape[0]
        a = env.action_space.n
        return s, a

    def __init__(self, agent, render=False):
        threading.Thread.__init__(self)
        self.render = render
        self.env = gym.make(ENV)
        self.agent = agent
        self.stop_signal = False

    def run(self):
        while not self.stop_signal:
            s = self.env.reset()
            R = 0
            while True:
                time.sleep(THREAD_DELAY)  # yield

                if self.render:
                    self.env.render()

                a = self.agent.act(s)
                s_, r, done, info = self.env.step(a)

                if done:  # terminal state
                    s_ = None

                self.agent.train(s, a, r, s_)

                s = s_
                R += r

                if done or self.stop_signal:
                    break

            print("Total R:", R)

    def stop(self):
        self.stop_signal = True
