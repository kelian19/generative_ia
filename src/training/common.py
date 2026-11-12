import torch

def device_for(cfg):
    device=torch.device(cfg.get('device','cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA demandé mais indisponible.')
    return device

def validate_training(loader,cfg):
    if cfg.get('epochs',0)<1 or not len(loader):raise ValueError('Au moins une époque et un batch sont nécessaires.')

def requires_grad(model,value):
    for parameter in model.parameters():parameter.requires_grad_(value)

def finite_loss(loss):
    if not torch.isfinite(loss):raise FloatingPointError('Perte non finie ; entraînement interrompu sans masquer le problème.')
