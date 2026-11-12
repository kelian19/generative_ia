import numpy as np
import torch
from scipy.special import inv_boxcox
from scipy.stats import ks_2samp,wasserstein_distance

def _inverse(scaled,scaler,lmbda,shift):
    transformed=scaler.inverse_transform(scaled)
    values=transformed if lmbda is None else inv_boxcox(transformed,lmbda)-shift
    if not np.isfinite(values).all():raise FloatingPointError('Inverse non finie : aucune simulation n’est supprimée ou tronquée.')
    return values

def _generate(model,latent_dim,scaler,lmbda,shift,n,seed,conditional=None,vae=False):
    if n<1:raise ValueError('Nombre de simulations positif requis.')
    device=next(model.parameters()).device;rng=torch.Generator(device=device).manual_seed(seed);was_training=model.training;model.eval()
    try:
        with torch.no_grad():
            z=torch.randn(n,latent_dim,device=device,generator=rng)
            if vae:
                scaled=model.decoder(z)
                scaled=scaled+torch.exp(model.log_scale.clamp(-6,3))*torch.randn(scaled.shape,device=device,generator=rng)
            elif conditional is not None:
                conditions=conditional.to(device)
                if not len(conditions):raise ValueError('Conditions vides.')
                idx=torch.randint(0,len(conditions),(n,),device=device,generator=rng);scaled=model(z,conditions[idx])
            else:scaled=model(z)
            return _inverse(scaled.detach().cpu().numpy(),scaler,lmbda,shift)
    finally:model.train(was_training)

def generate_vae_samples(model,latent_dim,scaler,lmbda,shift,n=10000,seed=100):return _generate(model,latent_dim,scaler,lmbda,shift,n,seed,vae=True)
def generate_gan_samples(generator,latent_dim,scaler,lmbda,shift,n=10000,seed=100):return _generate(generator,latent_dim,scaler,lmbda,shift,n,seed)
def generate_cgan_samples(generator,latent_dim,X_cond,scaler,lmbda,shift,n=10000,seed=100):return _generate(generator,latent_dim,scaler,lmbda,shift,n,seed,conditional=X_cond)

def finite_values(values):
    x=np.asarray(values,dtype=float).reshape(-1)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Échantillon fini et non vide requis.')
    return x

def compute_metrics(real,generated):
    a=finite_values(real);b=finite_values(generated);ks=ks_2samp(a,b)
    return {'wasserstein':float(wasserstein_distance(a,b)),'ks_stat':float(ks.statistic),'ks_pvalue_iid_diagnostic_only':float(ks.pvalue),
       'q99_real':float(np.quantile(a,.99)),'q99_gen':float(np.quantile(b,.99)),
       'var_99_real':float(np.quantile(a,.99)),'var_99_gen':float(np.quantile(b,.99)),
       'std_real':float(a.std()),'std_gen':float(b.std()),'max_real':float(a.max()),'max_gen':float(b.max())}
