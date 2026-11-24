"""Visuel 1080 × 1350 pour les réseaux, tiré des résultats versionnés (aucun chiffre saisi à la main).

python -m scripts.social_visual      (écrit docs/social_generatif.png)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.social_card import render,fr,ACCENT,WARN,SOFT,SAND,INK,MUTED
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
NETS=['vae','vae_weighted','gan','wgan_gp','cgan'];LAB={'empirical_train':'empirique','gaussian_train':'normale','vae':'VAE','vae_weighted':'VAE pond.','gan':'GAN','wgan_gp':'WGAN-GP','cgan':'cGAN'}

def main():
    ms=json.loads((R/'multiseed.json').read_text(encoding='utf-8'));pub=json.loads((R/'benchmark.json').read_text(encoding='utf-8'))
    d=pd.DataFrame(ms['per_seed']);agg=pd.DataFrame(ms['summary']);rep=ms['reproducibility']['max_abs_diff_wasserstein_vs_published_by_model']
    one=pub['models']['cgan']['test']['wasserstein'];gan1=pub['models']['gan']['test']['wasserstein'];cg=agg[(agg.model=='cgan')&(agg.split=='test')].iloc[0]
    def strip(ax):
        t=d[(d.split=='test')&d.model.isin(NETS)]
        for i,m in enumerate(NETS):
            v=t[t.model==m].wasserstein.to_numpy();c=WARN if m=='cgan' else ACCENT
            ax.scatter(np.full(len(v),i)+np.linspace(-0.12,0.12,len(v)),v,s=22,color=c,alpha=.8);ax.hlines(v.mean(),i-0.3,i+0.3,color=c,lw=2.2)
        ax.scatter([4],[one],s=110,facecolor='none',edgecolor=WARN,lw=1.6);ax.annotate('graine 100',(4,one),xytext=(-62,-2),textcoords='offset points',fontsize=8,color=WARN,va='center')
        ax.set_xticks(range(len(NETS)));ax.set_xticklabels([LAB[m] for m in NETS],fontsize=8);ax.set_ylabel('distance au test',fontsize=8)
    def valid(ax):
        v=agg[agg.split=='validation'].set_index('model').loc[['empirical_train','gan','vae','wgan_gp','vae_weighted','cgan']]
        cols=[ACCENT if m=='empirical_train' else SOFT for m in v.index]
        ax.bar(range(len(v)),v.w_mean,yerr=v.w_sd,color=cols,width=0.65,error_kw=dict(ecolor=INK,elinewidth=1,capsize=2))
        ax.set_xticks(range(len(v)));ax.set_xticklabels([LAB[m] for m in v.index],fontsize=7.5,rotation=20);ax.set_ylabel('distance (validation)',fontsize=8)
    n=len(ms['seeds'])
    cards=[('Le benchmark','VAE, VAE pondéré, GAN, WGAN-GP, cGAN\nsur l’indice de feux C-Haines (ERA5)',('big','5 réseaux',ACCENT,32)),
           ('Une graine','Les deux GAN en tête sur le test :\nGAN puis cGAN (distance, plus bas = mieux)',('big',f'{fr(gan1,2)} · {fr(one,2)}',WARN,40)),
           (f'{n} graines',f'cGAN : moyenne {fr(cg.w_mean,2)}, écart-type {fr(cg.w_sd,2)}\n(cercle : le résultat publié)',('draw',strip)),
           ('La référence','En validation, aucun réseau ne bat\nen moyenne le simple rééchantillonnage',('draw',valid)),
           ('Autre machine','Même graine, autre ordinateur :\nle cGAN bouge de',('big',fr(rep['cgan'],2),WARN,44)),
           ('La règle','Un écart entre modèles ne compte\nque s’il dépasse l’écart entre graines',('big','Tester plusieurs graines',ACCENT,22))]
    render(ROOT/'docs/social_generatif.png','IA GÉNÉRATIVE · RISQUE CLIMATIQUE','Un seul tirage peut\nfaire un champion.','Puis j’ai changé la graine aléatoire',cards,
           f'cGAN : {fr(one,2)} une fois, {fr(cg.w_mean,2)} en moyenne',
           'VAE · GAN · WGAN-GP · cGAN · ERA5  ·  github.com/kelian19/generative_ia')

if __name__=='__main__':main()
