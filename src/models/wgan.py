import torch
import torch.nn as nn
import torch.autograd as autograd


class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )
        # A Gaussian reference plus a trainable residual avoids a near-constant
        # initial generator; it imposes no target moments or tail inflation.
        nn.init.zeros_(self.model[-1].weight)
        nn.init.zeros_(self.model[-1].bias)

    def forward(self, z):
        return z[:, :1] + self.model(z)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


def compute_gradient_penalty(critic, real, fake):
    if real.shape != fake.shape or real.device != fake.device:
        raise ValueError('Real/fake doivent avoir mêmes forme et device.')
    alpha = torch.rand((real.size(0),)+(1,)*(real.ndim-1), device=real.device,dtype=real.dtype)
    interp=(alpha*real+(1-alpha)*fake).detach().requires_grad_(True)
    d_interp=critic(interp)
    grads=autograd.grad(d_interp,interp,torch.ones_like(d_interp),create_graph=True)[0]
    return ((grads.reshape(real.size(0),-1).norm(2,dim=1)-1)**2).mean()
