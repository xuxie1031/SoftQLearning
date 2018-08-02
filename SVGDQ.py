import numpy as np

import torch
import torch.nn as nn

from NNUtils import BaseNet

class SVGDQ(nn.Module, BaseNet):

    def __init__(self,
        state_dim,
        action_dim,
        hidden_sizes,
        hidden_nonlinearity=nn.ReLU,
        gpu=-1
    ):
        super(SVGDQ, self).__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.state_layer = nn.Linear(self.state_dim, hidden_sizes[0], bias=False)
        self.action_layer = nn.Linear(self.action_dim, hidden_sizes[0], bias=False)

        nn.init.xavier_uniform_(self.state_layer.weight)
        nn.init.xavier_uniform_(self.action_layer.weight)

        self.concate_layer_bias = nn.Parameter(torch.zeros(hidden_sizes[0]))

        self.rest_layers = [hidden_nonlinearity()]
        sizes = list(hidden_sizes)
        for idx, size in enumerate(sizes):
            if idx != len(sizes)-1:
                layer = nn.Linear(size, sizes[idx+1])
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
                self.rest_layers.append(layer)
                self.rest_layers.append(hidden_nonlinearity())

        layer = nn.Linear(sizes[len(sizes)-1], 1)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)
        self.rest_layers.append(layer)
        self.rest_layers = nn.Sequential(*self.rest_layers)

        self.set_gpu(gpu)


    def forward(self, state, action):
        state = self.tensor(state)
        action = self.tensor(action)

        inputs = [state, action]
        concate_layers = [self.state_layer, self.action_layer]

        result = self.concate_layer_bias
        for v, layer in zip(inputs, concate_layers):
            result = result+layer(v)
        
        result = self.rest_layers(result)

        return result


def main():
    qf = SVGDQ(3, 2, (100, 100), gpu=0)
    
    state = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    action = np.array([[2.0, 1.0], [3.0, 1.0]])
    print(qf.forward(state, action))

if __name__ == "__main__":
    main()