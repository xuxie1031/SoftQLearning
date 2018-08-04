from config import Config
from task import *

from SVGDPolicy import SVGDPolicy
from SVGDQ import SVGDQ
from SoftQAgent import SoftQAgent

import torch

def SoftQ():
    game = 'RoboschoolAnt-v1'
    config = Config()
    config.env = Roboschool(game)
    state_dim = config.env.state_dim
    action_dim = config.env.action_dim

    config.policy = SVGDPolicy(state_dim, action_dim, hidden_sizes=(100, 100))
    config.qf = SVGDQ(state_dim, action_dim, hidden_sizes=(100, 100))
    config.qf_target = SVGDQ(state_dim, action_dim, hidden_sizes=(100, 100))

    config.policy_optimizer = torch.optim.Adam
    config.qf_optimizer = torch.optim.Adam

    agent = SoftQAgent(config)
    agent.agent_episode()

if __name__ == '__main__':
    SoftQ()