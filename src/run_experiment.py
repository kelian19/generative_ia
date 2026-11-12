"""Comparaison reproductible, descriptive et non annuelle, sur dates chronologiques."""
import argparse,copy,hashlib,json,platform,time
from pathlib import Path
import importlib.metadata as md
import numpy as np
import pandas as pd
import torch
import config
from src.data_loader import load_haines
from src.preprocessing import build_dataframe,temporal_split,fit_transform,transform,make_loader
from src.training.train_vae import train_vae,train_vae_pro
from src.training.train_gan import train_gan
from src.training.train_wgan import train_wgan
from src.training.train_cgan import prepare_cgan_data,train_cgan
from src.evaluation import generate_vae_samples,generate_gan_samples,generate_cgan_samples,compute_metrics
from src.actuarial import compute_prime_pure
ROOT=Path(__file__).resolve().parents[1]

def run_experiment(epochs=10,n_samples=10000,seed=100,device='cpu',output='reports'):
    if epochs<2 or n_samples<100:raise ValueError('Au moins 2 époques et 100 simulations requis.')
    config.set_seed(seed);torch.set_num_threads(2)
    ds,index=load_haines(config.DATA_PATH);raw=build_dataframe(index)
    train,val,test=temporal_split(raw,gap_days=1)
    train,lmbda,scaler,shift=fit_transform(train);val=transform(val,scaler);test=transform(test,scaler)
    trigger=float(np.quantile(train.haines_index,.95))
    report={'source_repo':'https://github.com/Kelian2909/generative_ia','source_commit':'bca170c92297e2155c28663772c2bd9914987e1f',
       'data_sha256':hashlib.sha256(Path(config.DATA_PATH).read_bytes()).hexdigest(),'index_definition':index.attrs,
       'raw_grid_time_cells':int(index.size),'valid_grid_time_cells':len(raw),'dates_in_source':int(ds.sizes['valid_time']),
       'limitations':['Only July 2023; no annual or seasonal validation','Spatial and temporal dependence: iid KS p-values are diagnostic only','Independent scalar samples do not generate coherent atmospheric fields','No fire occurrence, loss or portfolio exposure data','RH outside ]0,100] masked; potential selection effect','No terrain mask for pressure levels below ground','Chronological test is local to the same month and locations; not future-climate validation'],
       'wgan_tuning':{'selection':'validation Wasserstein only','trials':['wgan_validation_trials_initial.json','wgan_penalty_trials.json'],'initial_failure_archive':'benchmark_initial.json','test_was_previously_seen':True,'final_test_role':'diagnostic, not a new independent evaluation'},
       'seed':seed,'epochs_budget_per_model':epochs,'n_samples':n_samples,'device':device,'split':{},'models':{},
       'trigger_demo':trigger,'trigger_origin':'95th percentile of TRAIN, demonstration only, not a contract threshold','indemnity_demo':config.INDEMNITE,
       'payment_unit':'uniform grid-cell/time observation; NOT annual premium','target_transform':{'type':'identity+StandardScaler','mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'fitted_on':'train'},
       'versions':{name:md.version(name) for name in ['torch','numpy','pandas','scipy','scikit-learn','xarray','netCDF4']}}
    for label,part in [('train',train),('validation',val),('test',test)]:
        report['split'][label]={'rows':len(part),'dates':int(part.valid_time.nunique()),'start':str(part.valid_time.min()),'end':str(part.valid_time.max()),'mean':float(part.haines_index.mean()),'q99':float(part.haines_index.quantile(.99))}
    start=time.perf_counter();rng=np.random.default_rng(seed)
    baseline_samples={'empirical_train':rng.choice(train.haines_index.to_numpy(),n_samples,replace=True),
        'gaussian_train':rng.normal(train.haines_index.mean(),train.haines_index.std(ddof=0),n_samples)}
    def evaluate(samples,part):
        result=compute_metrics(part.haines_index.to_numpy(),samples)
        result['trigger_probability'],result['expected_payment_per_observation']=compute_prime_pure(samples,trigger,config.INDEMNITE)
        result['observed_trigger_probability']=float((part.haines_index>trigger).mean())
        return result
    for name,samples in baseline_samples.items():report['models'][name]={'validation':evaluate(samples,val),'test':evaluate(samples,test),'trained':True,'baseline':True}
    for name,cfg0,trainer in [('vae',config.VAE,train_vae),('vae_weighted',config.VAE_PRO,train_vae_pro),('gan',config.GAN,train_gan),('wgan_gp',config.WGAN,train_wgan),('cgan',config.CGAN,train_cgan)]:
        config.set_seed(seed);cfg=copy.deepcopy(cfg0);cfg.update(epochs=epochs,device=device)
        if name=='vae_weighted':cfg.update(epochs=epochs//2,weighted_epochs=epochs-epochs//2)
        loader,_=make_loader(train,batch_size=config.BATCH_SIZE,seed=seed)
        then=time.perf_counter()
        if name=='cgan':
            loader,conditions,encoder=prepare_cgan_data(train,batch_size=config.BATCH_SIZE,seed=seed)
            model,_=trainer(loader,conditions,cfg)
            val_cond=torch.as_tensor(encoder.transform(val),dtype=torch.float32);test_cond=torch.as_tensor(encoder.transform(test),dtype=torch.float32)
            samples_val=generate_cgan_samples(model,cfg['latent_dim'],val_cond,scaler,lmbda,shift,n_samples,seed)
            samples_test=generate_cgan_samples(model,cfg['latent_dim'],test_cond,scaler,lmbda,shift,n_samples,seed)
        else:
            model,_=trainer(loader,cfg)
            generate=generate_vae_samples if name.startswith('vae') else generate_gan_samples
            samples_val=generate(model,cfg['latent_dim'],scaler,lmbda,shift,n_samples,seed);samples_test=samples_val
        report['models'][name]={'validation':evaluate(samples_val,val),'test':evaluate(samples_test,test),'config':cfg,'seconds':time.perf_counter()-then,'history':model.training_history_}
        artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True);torch.save(model.state_dict(),artifacts/(name+'.pt'))
        print(name,report['models'][name]['test'],flush=True)
    report['seconds']=time.perf_counter()-start
    folder=Path(output);folder=folder if folder.is_absolute() else ROOT/folder;folder.mkdir(parents=True,exist_ok=True)
    (folder/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (ROOT/'artifacts/preprocessing.json').write_text(json.dumps(report['target_transform'],indent=2))
    rows=[]
    for name,values in report['models'].items():
        for split in ['validation','test']:rows.append({'model':name,'split':split,**values[split]})
    pd.DataFrame(rows).to_csv(folder/'benchmark.csv',index=False)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--n-samples',type=int,default=10000);parser.add_argument('--seed',type=int,default=100);parser.add_argument('--device',default='cpu');parser.add_argument('--output',default='reports');args=parser.parse_args()
    run_experiment(args.epochs,args.n_samples,args.seed,args.device,args.output)
