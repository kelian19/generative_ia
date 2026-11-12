from pathlib import Path
import random
import numpy as np
import torch

SEED=100
DATA_PATH=str(Path(__file__).resolve().parent/'data/haines_data.nc')
BATCH_SIZE=128
N_SAMPLES=10000
# Must be chosen for the corrected index and contract; no inherited proxy threshold.
TRIGGER=None
INDEMNITE=10000

def set_seed(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

VAE=dict(latent_dim=2,lr=1e-3,epochs=10,kl_weight=1.,device='cpu')
VAE_PRO=dict(latent_dim=2,lr=5e-4,epochs=5,lr_weighted=1e-4,weighted_epochs=5,weight_scale=2.,kl_weight=1.,device='cpu')
GAN=dict(latent_dim=10,lr=.0002,epochs=10,device='cpu')
WGAN=dict(latent_dim=16,lr_g=.0001,lr_c=.001,epochs=10,n_critic=5,gp_lambda=.1,device='cpu')
CGAN=dict(latent_dim=10,lr=.0002,epochs=10,device='cpu',cond_features=['latitude','longitude','month_sin','month_cos'])
