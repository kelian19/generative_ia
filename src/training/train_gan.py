import torch
from src.models.gan import Generator,Discriminator
from .common import device_for,validate_training,finite_loss,requires_grad

def train_adversarial(train_loader,cfg,generator,discriminator,conditional=False):
    validate_training(train_loader,cfg);device=device_for(cfg);generator=generator.to(device);discriminator=discriminator.to(device)
    generator.train();discriminator.train()
    go=torch.optim.Adam(generator.parameters(),lr=cfg['lr'],betas=(.5,.999));do=torch.optim.Adam(discriminator.parameters(),lr=cfg['lr'],betas=(.5,.999))
    criterion=torch.nn.BCEWithLogitsLoss();history=[]
    for epoch in range(cfg['epochs']):
        dl=gl=0.;count=0
        for batch in train_loader:
            if conditional:cond,real=[t.to(device) for t in batch]
            else:real=batch[0].to(device);cond=None
            def discriminate(x):return discriminator(x,cond) if conditional else discriminator(x)
            def generate():
                z=torch.randn(len(real),cfg['latent_dim'],device=device)
                return generator(z,cond) if conditional else generator(z)
            requires_grad(discriminator,True);do.zero_grad(set_to_none=True)
            with torch.no_grad():fake=generate()
            loss_d=.5*(criterion(discriminate(real),torch.ones_like(real))+criterion(discriminate(fake),torch.zeros_like(real)))
            finite_loss(loss_d);loss_d.backward();do.step()
            requires_grad(discriminator,False);go.zero_grad(set_to_none=True)
            loss_g=criterion(discriminate(generate()),torch.ones_like(real));finite_loss(loss_g);loss_g.backward();go.step()
            dl+=loss_d.item()*len(real);gl+=loss_g.item()*len(real);count+=len(real)
        history.append({'epoch':epoch+1,'discriminator_loss':dl/count,'generator_loss':gl/count})
    requires_grad(discriminator,True);generator.training_history_=history
    return generator,discriminator

def train_gan(train_loader,cfg):return train_adversarial(train_loader,cfg,Generator(cfg['latent_dim']),Discriminator())
