class Config:
    def __init__(self):
        self.env = None
        self.policy = None
        self.qf = None
        self.qf_target = None
        self.policy_optimizer = None
        self.qf_optimizer = None
        self.batch_size = 16
        self.n_episodes = 10000
        self.min_replay_size = 10000
        self.replay_size = 1000000
        self.discount = .99
        self.alpha = 1.0
        self.soft_target_tau = 1000
        self.kernel_K = 8
        self.kernel_update_ratio = .5
        self.qf_target_K = 4
        self.policy_learning_rate = 1e-4
        self.qf_learning_rate = 1e-4