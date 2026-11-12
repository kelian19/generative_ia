"""VAE continu à vraisemblance Gaussienne, decoder non borné et bruit d'observation."""
import math
import torch
import torch.nn as nn

class HainesVAE(nn.Module):
    def __init__(self,input_dim=1,latent_dim=2,hidden=(32,16)):
        super().__init__();a,b=hidden
        self.encoder=nn.Sequential(nn.Linear(input_dim,a),nn.ReLU(),nn.Linear(a,b),nn.ReLU())
        self.fc_mu=nn.Linear(b,latent_dim);self.fc_logvar=nn.Linear(b,latent_dim)
        self.decoder=nn.Sequential(nn.Linear(latent_dim,b),nn.ReLU(),nn.Linear(b,a),nn.ReLU(),nn.Linear(a,input_dim))
        self.log_scale=nn.Parameter(torch.full((input_dim,),-1.))
    def reparameterize(self,mu,logvar):return mu+torch.exp(.5*logvar)*torch.randn_like(mu)
    def forward(self,x):
        h=self.encoder(x);mu=self.fc_mu(h);logvar=self.fc_logvar(h).clamp(-20,20)
        return self.decoder(self.reparameterize(mu,logvar)),mu,logvar

class HainesVAE_Pro(HainesVAE):
    def __init__(self,input_dim=1,latent_dim=2):super().__init__(input_dim,latent_dim,hidden=(64,32))

def elbo_loss(recon_x,x,mu,logvar,log_scale=None,beta=1.):
    if log_scale is None:log_scale=torch.zeros((),device=x.device,dtype=x.dtype)
    log_scale=log_scale.clamp(-6,3)
    nll=.5*((x-recon_x)*torch.exp(-log_scale))**2+log_scale+.5*math.log(2*math.pi)
    kl=-.5*torch.sum(1+logvar-mu.square()-logvar.exp(),dim=1)
    return torch.sum(nll)+beta*kl.sum()

def weighted_elbo_loss(recon_x,x,mu,logvar,weight_scale=2.,log_scale=None,beta=1.):
    # Intentional change in reconstruction objective, NOT a calibrated tail likelihood.
    if weight_scale<0:raise ValueError('weight_scale doit être positif.')
    if log_scale is None:log_scale=torch.zeros((),device=x.device,dtype=x.dtype)
    weight=torch.exp(torch.sigmoid(x)*min(float(weight_scale),10.));weight=weight/weight.mean().detach()
    log_scale=log_scale.clamp(-6,3)
    nll=.5*((x-recon_x)*torch.exp(-log_scale))**2+log_scale+.5*math.log(2*math.pi)
    kl=-.5*torch.sum(1+logvar-mu.square()-logvar.exp())
    return (weight*nll).sum()+beta*kl
