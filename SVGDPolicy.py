import numpy as np
import torch

class SVGDPolicy(Object):

	def __init__(
		self,
		state_dim,
		action_dim,
		)