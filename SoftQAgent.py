import numpy as np

import torch
import torch.nn as nn
from torch import optim

from Replay import Replay
from AdaptiveIsotropicGaussianKernel import AdaptiveIsotropicGaussianKernel

class SoftQAgent:
    def __init__(self, config): # initialize with config variables
        self.env = config.env
        self.state_dim = int(config.env.state_dim)
        self.action_dim = int(config.env.action_dim)
        self.policy = config.policy
        self.qf = config.qf
        self.qf_target = config.qf_target
        self.batch_size = config.batch_size
        self.n_episodes = config.n_episodes
        self.min_replay_size = config.min_replay_size
        self.replay_size = config.replay_size
        self.discount = config.discount
        self.soft_target_tau = config.soft_target_tau
        self.alpha = config.alpha

        self.kernel_K = config.kernel_K
        self.kernel_update_ratio = config.kernel_update_ratio
        self.kernel_K_updated = int(self.kernel_K*self.kernel_update_ratio)
        self.kernel_K_fixed = self.kernel_K-self.kernel_K_updated

        self.qf_target_K = config.qf_target_K
        self.qf_target.load_state_dict(self.qf.state_dict())

        self.qf_optimizer = config.qf_optimizer(self.qf.parameters(), lr=config.qf_learning_rate)
        self.policy_optimizer = config.policy_optimizer(self.policy.parameters(), lr=config.policy_learning_rate)

        self.episode_rewards = []
        self.total_steps = 0


    def agent_episode(self):    # agent training
        replay = Replay(self.replay_size, self.batch_size)

        for episode in range(self.n_episodes):

            rewards = 0
            state = self.env.reset()
            while True:
                action = self.policy.sample_action(np.stack([state]))
                next_state, reward, terminal, _ = self.env.step(action)
                replay.feed([state, action, reward, next_state, int(terminal)])

                self.total_steps += 1
                rewards += reward
                state = next_state

                if replay.size() >= self.min_replay_size:
                    experiences = replay.sample()
                    self.agent_training(experiences)

                if self.total_steps % self.soft_target_tau == 0:
                    self.qf_target.load_state_dict(self.qf.state_dict())

                if terminal:
                    self.episode_rewards.append(rewards)
                    # print training info ...
                    print('epsisode %d total step %d avg reward %f' % (episode, self.total_steps, np.mean(np.array(self.episode_rewards[-100:]))))
                    break


    def agent_training(self, experiences):
        self.train_qf(experiences, 1.0)
        self.train_policy(experiences[0])


    def train_qf(self, experiences, action_max):
        def log_sum_exp(value, dim):
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            m = m.squeeze(dim)
            return m+torch.log(torch.sum(torch.exp(value0), dim=dim))


        states, actions, rewards, next_states, terminals = experiences

        states = self.qf.tensor(states)     # N*state_dim
        actions = self.qf.tensor(actions)   # N*action_dim
        rewards = self.qf.tensor(rewards)
        next_states = self.qf.tensor(next_states)
        terminals = self.qf.tensor(terminals)

        clipped_actions = torch.clamp(actions, -action_max, action_max)
        q_curr = self.qf(states, clipped_actions)   # N*1
        q_curr = q_curr.squeeze()   # N

        next_states_expanded = next_states.unsqueeze(1)  # N*1*state_dim
        target_actions = np.random.uniform(-1.0, 1.0, size=(1, self.qf_target_K, self.action_dim))  # 1*self.qf_target_K*action_dim
        q_next = self.qf_target(next_states_expanded, target_actions)   # N*self.qf_target_K*1

        v_next = log_sum_exp(q_next, 1).squeeze()   # N
        v_next = v_next-torch.log(self.qf.tensor([self.qf_target_K]))
        # v_next = v_next+self.action_dim*torch.log(self.qf.tensor([2.0]))    # purpose?

        ys = rewards+(1.0-terminals)*self.discount*v_next   # N
        ys = ys.detach()

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_curr, ys)  # 1

        self.qf_optimizer.zero_grad()
        loss.backward()
        self.qf_optimizer.step()

    
    def train_policy(self, states):
        states = self.policy.tensor(states)   # N*state_dim
        actions_fixed = self.policy(states, self.kernel_K_fixed)    # N*K_fix*action_dim
        actions_updated = self.policy(states, self.kernel_K_updated)    # N*K_update*action_dim
        
        def invert_grad(actions, x):
            invert_max_v = 10.0
            greater_idx = actions > 1.0
            lower_idx = actions < -1.0
            x[greater_idx] = -invert_max_v
            x[lower_idx] = invert_max_v
            return x
        actions_hooked = actions_fixed.detach()
        actions_fixed.register_hook(lambda x: invert_grad(actions_hooked, x))

        states_expanded = states.unsqueeze(1)   # N*1*state_dim
        q_unbounded = self.qf(states_expanded, actions_fixed)   # N*K_fix*1

        grad_q_action_fixed = torch.autograd.grad(q_unbounded, actions_fixed, grad_outputs=torch.ones(q_unbounded.size(), device=self.policy.device))[0]   # N*K_fix*action_dim
        # grad_q_action_fixed = grad_q_action_fixed.unsqueeze(2).detach()     # N*K_fix*1*action_dim

        # # kernel calculation
        # kernel = AdaptiveIsotropicGaussianKernel(actions_fixed, actions_updated, self.policy.tensor([1e-5]))
        # kappa = kernel.kappa    # N*K_fix*K_update
        # kappa = kappa.unsqueeze(3)     # N*K_fix*K_update*1
        # kappa_grads = kernel.kappa_grad   # N*K_fix*K_update*action_dim

        # # print(grad_q_action_fixed)

        # action_grads = torch.mean(kappa*grad_q_action_fixed+self.alpha*kappa_grads, dim=1)  # N*K_update*action_dim  

        actions_loss = -actions_updated
        self.policy_optimizer.zero_grad()
        # actions_loss.backward(gradient=action_grads)
        actions_loss.backward(gradient=grad_q_action_fixed)
        self.policy_optimizer.step()
