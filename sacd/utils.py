from collections import deque
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:
    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
