import torch
import torch.nn as nn

class BaseNet:

    def set_gpu(self, gpu):
        if gpu >=0 and torch.cuda.is_available():
            gpu = gpu % torch.cuda.device_count()
            self.device = torch.device('cuda:%d' % (gpu))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)
    

    def tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return x