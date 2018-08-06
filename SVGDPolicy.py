import numpy as np
import torch
import torch.nn as nn

from NNUtils import BaseNet

'''
SVGDPolicy setting with series of linear layers,
which dims are specified in hidden_sizes
'''

class SVGDPolicy(nn.Module, BaseNet):

	def __init__(
		self,
		state_dim,
		action_dim,
		hidden_sizes,
		hidden_nonlinearity=nn.ReLU,
		output_nonlinearity=None,
		gpu=-1
	):
		super(SVGDPolicy, self).__init__()
		self.state_dim = int(state_dim)
		self.action_dim = int(action_dim)

		self.state_layer = nn.Linear(self.state_dim, hidden_sizes[0], bias=False)
		self.noise_layer = nn.Linear(self.action_dim, hidden_sizes[0], bias=False)
		nn.init.xavier_uniform_(self.state_layer.weight)
		nn.init.xavier_uniform_(self.noise_layer.weight)

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
		
		layer = nn.Linear(sizes[len(sizes)-1], self.action_dim)
		nn.init.xavier_uniform_(layer.weight)
		nn.init.constant_(layer.bias, 0)
		self.rest_layers.append(layer)

		if output_nonlinearity is not None:
			self.rest_layers.append(output_nonlinearity())
		self.rest_layers = nn.Sequential(*self.rest_layers)

		self.set_gpu(gpu)
	

	def forward(self, state, k):
		state = self.tensor(state)
		sample_shape = (1, k, self.action_dim)

		expanded_inputs = []
		state_expanded = state.unsqueeze(1)	# N*1*state_dim
		expanded_inputs.append(state_expanded)
		epsilon = np.random.randn(*sample_shape) # 1*k*action_dim
		epsilon = self.tensor(epsilon)
		expanded_inputs.append(epsilon)

		concate_layers = [self.state_layer, self.noise_layer]
		result = self.concate_layer_bias
		for v, layer in zip(expanded_inputs, concate_layers):
			result = result + layer(v)
		result = self.rest_layers(result)	# N*k*action_dim

		return result

	
	def sample_action(self, state):
		action = self.forward(state, k=1)	# 1*action_dim
		action = action[0][0]	# only pick up one action from batch

		return action.cpu().detach().numpy()


def main():
	policy = SVGDPolicy(state_dim=3, action_dim=2, hidden_sizes=(100, 100), output_nonlinearity=None, gpu=0)
	state = np.array([[1.0, 2.0, 3.0]])
	print(policy.sample_action(state))

if __name__ == '__main__':
	main()