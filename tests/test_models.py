import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
import config
from src.models.vae import HainesVAE,elbo_loss,weighted_elbo_loss
from src.models.gan import Generator
from src.models.wgan import compute_gradient_penalty
from src.preprocessing import make_loader
from src.training.train_gan import train_gan
from src.training.train_vae import train_vae,train_vae_pro
from src.training.train_wgan import train_wgan
from src.training.train_cgan import ConditionEncoder,prepare_cgan_data,train_cgan
from src.evaluation import generate_vae_samples,generate_gan_samples

def test_generator_and_decoder_are_not_bounded_by_training_extrema():
 g=Generator(2)
 with torch.no_grad():g.model[-1].weight.zero_();g.model[-1].bias.fill_(10.)
 s=StandardScaler().fit(np.array([[0.],[1.],[2.]]))
 samples=generate_gan_samples(g,2,s,None,0,n=10)
 assert samples.min()>2
 assert not isinstance(HainesVAE().decoder[-1],torch.nn.Sigmoid)

def test_gaussian_decoder_noise_is_sampled_reproducibly():
 m=HainesVAE()
 with torch.no_grad():
  for p in m.decoder.parameters():p.zero_()
  m.log_scale.fill_(0.)
 s=StandardScaler().fit([[-1.],[1.]])
 a=generate_vae_samples(m,2,s,None,0,n=500,seed=9);b=generate_vae_samples(m,2,s,None,0,n=500,seed=9)
 np.testing.assert_equal(a,b);assert a.std()>.8
 assert m.training

def test_gradient_penalty_has_known_value_and_correct_dtype():
 critic=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
 with torch.no_grad():critic.weight.fill_(2.)
 real=torch.randn(4,1,dtype=torch.float64);fake=torch.randn_like(real)
 gp=compute_gradient_penalty(critic,real,fake)
 assert gp.item()==pytest.approx(1.)
 gp.backward();assert critic.weight.grad.isfinite().all()

def test_elbo_is_gaussian_and_weighted_objective_stays_finite():
 x=torch.zeros(2,1);mu=torch.zeros(2,2);lv=torch.zeros_like(mu)
 assert elbo_loss(x,x,mu,lv).item()==pytest.approx(np.log(2*np.pi))
 loss=weighted_elbo_loss(torch.zeros(2,1),torch.tensor([[100.],[-100.]]),mu,lv,weight_scale=5)
 assert torch.isfinite(loss)

def dataset():
 return pd.DataFrame({'haines_final':np.linspace(-1,1,17),'valid_time':pd.date_range('2023-07-01',periods=17),'latitude':np.linspace(41,51,17),'longitude':np.linspace(-5,10,17)})

@pytest.mark.parametrize('method',['vae','weighted','gan','wgan','cgan'])
def test_each_training_loop_updates_parameters_with_last_batch_of_one(method):
 config.set_seed();torch.set_num_threads(1);df=dataset();loader,_=make_loader(df,batch_size=8)
 cfg=dict(latent_dim=2,lr=.001,epochs=2,device='cpu',lr_g=.001,lr_c=.001,n_critic=2,gp_lambda=10.,weighted_epochs=1,weight_scale=2.)
 if method=='vae':m,_=train_vae(loader,cfg)
 elif method=='weighted':m,_=train_vae_pro(loader,cfg)
 elif method=='gan':m,_=train_gan(loader,cfg)
 elif method=='wgan':
  m,_=train_wgan(loader,cfg)
  assert sum(row['generator_updates'] for row in m.training_history_)==3
 else:
  cond_loader,conditions,_=prepare_cgan_data(df,batch_size=8)
  m,_=train_cgan(cond_loader,conditions,cfg)
 assert m.training_history_
 assert all(p.isfinite().all() for p in m.parameters())
 # At least one parameter must have actually received a finite gradient.
 assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in m.parameters())

def test_conditional_encoding_is_cyclic_and_rejects_unknown_months():
 df=dataset();encoder=ConditionEncoder().fit(df);values=encoder.transform(df)
 assert values.shape==(17,4)
 np.testing.assert_allclose(values[:,2]**2+values[:,3]**2,1)
 outside=df.copy();outside.valid_time=pd.Timestamp('2023-12-01')
 with pytest.raises(ValueError,match='Mois'):encoder.transform(outside)

def test_seed_reproduces_gan_training_on_cpu():
 states=[]
 for _ in range(2):
  config.set_seed(123);torch.set_num_threads(1)
  loader,_=make_loader(dataset(),batch_size=8,seed=123)
  model,_=train_gan(loader,dict(latent_dim=2,lr=.001,epochs=1,device='cpu'))
  states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
 for key in states[0]:torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)

def test_cgan_rejects_unseen_geographic_domain():
 df=dataset();encoder=ConditionEncoder().fit(df);outside=df.copy();outside.latitude=90
 with pytest.raises(ValueError,match='Coordonnées'):encoder.transform(outside)
