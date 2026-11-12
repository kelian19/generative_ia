"""Prétraitement appris uniquement sur train et séparation par dates, pas par lignes."""
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset

def build_dataframe(haines_index):
    df=haines_index.to_dataframe(name='haines_index').reset_index()
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=['haines_index','valid_time','latitude','longitude'])
    # Diagnostic labels only, not validated probability/severity categories.
    df['risk_level']=pd.cut(df.haines_index,[-np.inf,7,9,np.inf],labels=['Low','Moderate','High'],right=False)
    return df

def temporal_split(df,train_fraction=.6,validation_fraction=.2,gap_days=1):
    if not 0<train_fraction<1 or not 0<validation_fraction<1 or train_fraction+validation_fraction>=1:raise ValueError('Fractions train/validation invalides.')
    if not isinstance(gap_days,int) or gap_days<0:raise ValueError('Embargo entier positif ou nul requis.')
    times=np.sort(pd.to_datetime(df.valid_time).unique())
    if len(times)<10:raise ValueError('Au moins dix dates distinctes sont requises pour cette séparation chronologique.')
    a=int(len(times)*train_fraction);b=a+int(len(times)*validation_fraction)
    train_times=times[:a]
    val_start=pd.Timestamp(times[a])+pd.Timedelta(days=gap_days)
    test_start=pd.Timestamp(times[b])+pd.Timedelta(days=gap_days)
    selected=[train_times,times[(times>=val_start)&(times<times[b])],times[times>=test_start]]
    output=tuple(df.loc[pd.to_datetime(df.valid_time).isin(t)].copy() for t in selected)
    if any(part.empty for part in output):raise ValueError('Embargo trop long pour les dates disponibles.')
    return output

def fit_transform(df):
    result=df.copy();values=result[['haines_index']].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Cible finie et non vide requise.')
    scaler=StandardScaler().fit(values)
    if scaler.var_[0]<=0:raise ValueError('Cible constante : aucun modèle génératif identifiable.')
    result['haines_transformed']=values[:,0];result['haines_final']=scaler.transform(values)[:,0]
    # Tuple retained for compatibility; lambda=None denotes identity, no Box-Cox domain.
    return result,None,scaler,0.

def transform(df,scaler):
    result=df.copy();values=result[['haines_index']].to_numpy(dtype=float)
    if not np.isfinite(values).all():raise ValueError('Cible non finie.')
    result['haines_transformed']=values[:,0];result['haines_final']=scaler.transform(values)[:,0]
    return result

def print_stats(df):
    x=df.haines_index.to_numpy(dtype=float)
    return {'n':len(x),'skewness':float(stats.skew(x)),'kurtosis':float(stats.kurtosis(x))}

def make_loader(df,batch_size=128,seed=100,shuffle=True):
    if not len(df):raise ValueError('Données d’entraînement vides.')
    x=torch.as_tensor(df.haines_final.to_numpy(dtype=np.float32)).reshape(-1,1)
    loader=DataLoader(TensorDataset(x),batch_size=batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(seed))
    return loader,x
