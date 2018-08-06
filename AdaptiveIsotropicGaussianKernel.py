import numpy as np
import torch

class AdaptiveIsotropicGaussianKernel:
    def __init__(self, xs, ys, h_min=1e-3):
        self.xs = xs
        self.ys = ys
        self.h_min = h_min

        Kx, D1 = self.xs.size()[-2:]
        Ky, D2 = self.ys.size()[-2:]
        assert D1 == D2

        leading_shape = self.xs.size()[:-2]

        diff = self.xs.unsqueeze(-2)-self.ys.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(dim=-1, keepdim=False)

        shape = leading_shape+torch.Size([Kx*Ky])
        v, _ = dist_sq.view(shape).topk(k=(Kx*Ky//2+1))

        medians_sq = v[..., -1]

        h = medians_sq/np.log(Kx)
        h = torch.max(h, self.h_min)
        h = h
        h_expanded_twice = h.unsqueeze(-1).unsqueeze(-1)

        kappa = torch.exp(-dist_sq/h_expanded_twice)
        h_expanded_thrice = h_expanded_twice.unsqueeze(-1)
        kappa_expanded = kappa.unsqueeze(-1)
        kappa_grad = -2*diff/h_expanded_thrice*kappa_expanded

        self.kappa = kappa
        self.kappa_grad = kappa_grad