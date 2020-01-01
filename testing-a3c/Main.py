import time
from A3C.Brain import Brain
from A3C.Environment import Environment
from A3C.Optimizer import Optimizer
from A3C.Agent import Agent
from A3C.ThreadsManager import ThreadsManager
from A3C.Config import *


brain = Brain()
envs = ThreadsManager([Environment(Agent(brain)) for _ in range(THREADS)])
opts = ThreadsManager([Optimizer(brain) for _ in range(OPTIMIZERS)])

with envs, opts:
    time.sleep(RUN_TIME)

print("Training finished")
env_test = Environment(Agent(brain, use_epsilon=False), render=True)
env_test.run()
