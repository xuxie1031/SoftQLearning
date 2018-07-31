import torch
import torch.nn

from NNUtils import BaseNet

class SVGDQ(nn.Module, BaseNet):

    def __init__(self,
        state_dim,
        action_dim,
        hidden_sizes,
        hidden_nonlinearity=nn.ReLU
    ):
        super(SVGDQ, self).__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        