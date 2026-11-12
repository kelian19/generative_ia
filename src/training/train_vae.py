import torch
from src.models.vae import HainesVAE,HainesVAE_Pro,elbo_loss,weighted_elbo_loss
from .common import device_for,validate_training,finite_loss

def _train(model,loader,cfg,weighted=False):
    validate_training(loader,cfg);device=device_for(cfg);model=model.to(device);model.train();history=[]
    phases=[('standard',cfg['epochs'],cfg['lr'])]
    if weighted:phases.append(('weighted',cfg.get('weighted_epochs',0),cfg.get('lr_weighted',cfg['lr'])))
    for phase,epochs,lr in phases:
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        for epoch in range(epochs):
            total=0.;count=0
            for (x,) in loader:
                x=x.to(device);opt.zero_grad(set_to_none=True);recon,mu,lv=model(x)
                kwargs={'log_scale':model.log_scale,'beta':cfg.get('kl_weight',1.)}
                loss=weighted_elbo_loss(recon,x,mu,lv,weight_scale=cfg.get('weight_scale',2.),**kwargs) if phase=='weighted' else elbo_loss(recon,x,mu,lv,**kwargs)
                loss=loss/len(x);finite_loss(loss);loss.backward();opt.step();total+=loss.item()*len(x);count+=len(x)
            history.append({'phase':phase,'epoch':epoch+1,'loss_per_observation':total/count})
    model.training_history_=history
    return model,[row['loss_per_observation'] for row in history]

def train_vae(train_loader,cfg):return _train(HainesVAE(latent_dim=cfg['latent_dim']),train_loader,cfg)
def train_vae_pro(train_loader,cfg):return _train(HainesVAE_Pro(latent_dim=cfg['latent_dim']),train_loader,cfg,weighted=True)
