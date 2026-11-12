import torch
from src.models.wgan import Generator,Critic,compute_gradient_penalty
from .common import device_for,validate_training,finite_loss,requires_grad

def train_wgan(train_loader,cfg):
    validate_training(train_loader,cfg);device=device_for(cfg)
    gen=Generator(cfg['latent_dim']).to(device);critic=Critic().to(device)
    go=torch.optim.Adam(gen.parameters(),lr=cfg['lr_g'],betas=(0.,.9));co=torch.optim.Adam(critic.parameters(),lr=cfg['lr_c'],betas=(0.,.9))
    n_critic=cfg.get('n_critic',5)
    if n_critic<1:raise ValueError('n_critic doit être positif.')
    steps=0;history=[]
    for epoch in range(cfg['epochs']):
        critic_loss=generator_loss=0.;critic_updates=generator_updates=0
        for (real,) in train_loader:
            real=real.to(device);requires_grad(critic,True);co.zero_grad(set_to_none=True)
            with torch.no_grad():fake=gen(torch.randn(len(real),cfg['latent_dim'],device=device))
            gp=compute_gradient_penalty(critic,real,fake)
            loss_c=critic(fake).mean()-critic(real).mean()+cfg.get('gp_lambda',10.)*gp
            finite_loss(loss_c);loss_c.backward();co.step();steps+=1;critic_updates+=1;critic_loss+=loss_c.item()
            if steps%n_critic==0:
                requires_grad(critic,False);go.zero_grad(set_to_none=True)
                # Canonical WGAN loss: no reciprocal-std term that invents tail dispersion.
                loss_g=-critic(gen(torch.randn(len(real),cfg['latent_dim'],device=device))).mean()
                finite_loss(loss_g);loss_g.backward();go.step();generator_updates+=1;generator_loss+=loss_g.item()
        history.append({'epoch':epoch+1,'critic_loss':critic_loss/critic_updates,'generator_loss':generator_loss/generator_updates if generator_updates else None,'critic_updates':critic_updates,'generator_updates':generator_updates})
    if not any(row['generator_updates'] for row in history):raise ValueError('Budget insuffisant pour une mise à jour du générateur avec n_critic choisi.')
    requires_grad(critic,True);gen.training_history_=history
    return gen,critic
