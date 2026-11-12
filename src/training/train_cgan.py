import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from src.models.cgan import Generator,Discriminator
from .train_gan import train_adversarial

class ConditionEncoder:
    def fit(self,df):
        self.scaler_=StandardScaler().fit(df[['latitude','longitude']].to_numpy(dtype=float))
        self.bounds_=df[['latitude','longitude']].agg(['min','max'])
        self.months_=set(pd.to_datetime(df.valid_time).dt.month.unique())
        return self
    def transform(self,df):
        months=pd.to_datetime(df.valid_time).dt.month.to_numpy()
        if not set(months).issubset(self.months_):raise ValueError('Mois absent de l’apprentissage : saisonnalité hors domaine.')
        raw_coords=df[['latitude','longitude']]
        if ((raw_coords<self.bounds_.loc['min'])|(raw_coords>self.bounds_.loc['max'])).any().any():raise ValueError('Coordonnées hors du domaine d’apprentissage.')
        coords=self.scaler_.transform(df[['latitude','longitude']].to_numpy(dtype=float))
        out=np.column_stack([coords,np.sin(2*np.pi*months/12),np.cos(2*np.pi*months/12)])
        if not np.isfinite(out).all():raise ValueError('Conditions non finies.')
        return out

def prepare_cgan_data(df,batch_size=128,scaler_cond=None,seed=100):
    encoder=ConditionEncoder().fit(df) if scaler_cond is None else scaler_cond
    cond=torch.as_tensor(encoder.transform(df),dtype=torch.float32)
    y=torch.as_tensor(df.haines_final.to_numpy(dtype=np.float32)).reshape(-1,1)
    loader=DataLoader(TensorDataset(cond,y),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    return loader,cond,encoder

def train_cgan(train_loader,X_cond,cfg):
    return train_adversarial(train_loader,cfg,Generator(cfg['latent_dim'],X_cond.shape[1]),Discriminator(X_cond.shape[1]),conditional=True)
