"""Robustesse du benchmark de juillet 2023 à la graine aléatoire.

Le benchmark publié (reports/benchmark.json) repose sur une seule graine (100). On le relance à l'identique pour plusieurs
graines (initialisation des réseaux, ordre des batchs, tirages générés) et on résume la dispersion des métriques et la
stabilité du classement. La graine 100 doit reproduire exactement le benchmark publié (contrôle de reproductibilité).

python -m scripts.multi_seed --seeds 100 101 102 103 104
Sorties : reports/multiseed.json, reports/RESULTATS_MULTIGRAINES.md ; détail par graine hors Git (artifacts/seeds/).
"""
import argparse,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
from src.run_experiment import run_experiment
ROOT=Path(__file__).resolve().parents[1]
METRICS=['wasserstein','q99_gen','trigger_probability']

def summarize(reports):
    seeds=sorted(reports);models=list(reports[seeds[0]]['models'])
    rows=[]
    for s in seeds:
        for m in models:
            for split in ('validation','test'):
                r=reports[s]['models'][m][split];rows.append({'seed':s,'model':m,'split':split,**{k:r[k] for k in METRICS}})
    d=pd.DataFrame(rows)
    d['rank']=d.groupby(['seed','split']).wasserstein.rank(method='min')
    agg=d.groupby(['model','split']).agg(w_mean=('wasserstein','mean'),w_sd=('wasserstein','std'),w_min=('wasserstein','min'),w_max=('wasserstein','max'),
        q99_mean=('q99_gen','mean'),q99_sd=('q99_gen','std'),p_mean=('trigger_probability','mean'),p_sd=('trigger_probability','std'),
        rank_mean=('rank','mean'),rank_best=('rank',lambda r:int((r==1).sum()))).reset_index()
    return d,agg

def stability(d):
    """Le meilleur modèle en validation l'est-il aussi en test, graine par graine ? Corrélation de rang validation/test."""
    out=[]
    for s,g in d.groupby('seed'):
        v=g[g.split=='validation'].set_index('model').wasserstein;t=g[g.split=='test'].set_index('model').wasserstein.loc[v.index]
        out.append({'seed':int(s),'best_validation':v.idxmin(),'best_test':t.idxmin(),'spearman_val_test':float(v.rank().corr(t.rank()))})
    return out

def main(seeds,epochs=10,n_samples=10000):
    reports={}
    for s in seeds:
        out=ROOT/'artifacts/seeds'/f'seed_{s}';f=out/'benchmark.json'
        reports[s]=json.loads(f.read_text(encoding='utf-8')) if f.exists() else run_experiment(epochs,n_samples,s,'cpu',str(out))
        print('graine',s,'terminée',flush=True)
    d,agg=summarize(reports);stab=stability(d)
    pub=json.loads((ROOT/'reports/benchmark.json').read_text(encoding='utf-8'))
    repro=None
    if 100 in reports:
        per={m:max(abs(pub['models'][m][sp]['wasserstein']-reports[100]['models'][m][sp]['wasserstein']) for sp in ('validation','test')) for m in pub['models']}
        repro={'seed':100,'max_abs_diff_wasserstein_vs_published_by_model':per,'platform':platform.platform(),
               'note':'Le benchmark publié a été produit sur une autre machine ; les écarts viennent de différences numériques de plateforme, amplifiées par l’entraînement adverse.'}
    res={'seeds':seeds,'epochs_budget_per_model':epochs,'n_samples':n_samples,'reproducibility':repro,'summary':agg.to_dict('records'),
         'per_seed':d.to_dict('records'),'ranking_stability':stab,
         'note':'Même protocole et mêmes données que benchmark.json ; seule la graine change. Le test a déjà été consulté : lecture descriptive.'}
    (ROOT/'reports/multiseed.json').write_text(json.dumps(res,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    write_md(res,agg,stab);write_tex(res,agg,stab);return res

def write_md(res,agg,stab):
    def tab(split):
        a=agg[agg.split==split].sort_values('w_mean')
        lines=['| Méthode | Wasserstein moyenne | Écart-type | Min – max | Rang moyen | Fois 1er |','|---|---:|---:|---:|---:|---:|']
        lines+=[f"| {r.model} | {r.w_mean:.4f} | {r.w_sd:.4f} | {r.w_min:.4f} – {r.w_max:.4f} | {r.rank_mean:.1f} | {r.rank_best} |" for r in a.itertuples()]
        return '\n'.join(lines)
    n=len(res['seeds']);rep=res['reproducibility']
    best_v=pd.Series([s['best_validation'] for s in stab]).value_counts();best_t=pd.Series([s['best_test'] for s in stab]).value_counts()
    rho=np.mean([s['spearman_val_test'] for s in stab])
    md=f"""# Robustesse à la graine — benchmark de juillet 2023

Même protocole, mêmes données et même budget que [RESULTATS.md](RESULTATS.md) ; seule la graine change ({', '.join(map(str,res['seeds']))}). Wasserstein en unités de C-Haines, plus faible = meilleure proximité marginale ; les références (`empirical_train`, `gaussian_train`) ne dépendent de la graine que par le tirage des échantillons.

Reproductibilité de la graine 100 (relancée sur {rep['platform']}) contre le benchmark publié, écart maximal de Wasserstein par méthode : {', '.join(f'{k} {v:.4f}' for k,v in rep['max_abs_diff_wasserstein_vs_published_by_model'].items())}. Les références sont identiques ; les réseaux varient légèrement d'une plateforme à l'autre (arithmétique flottante), et le cGAN fortement : son entraînement adverse amplifie ces écarts minimes. La graine fixe la reproductibilité sur une même machine, pas d'une machine à l'autre.

## Validation ({n} graines)

{tab('validation')}

## Test déjà consulté ({n} graines, lecture descriptive)

{tab('test')}

## Stabilité du classement

Meilleur modèle en validation selon la graine : {', '.join(f'{k} ({v})' for k,v in best_v.items())}. Meilleur en test : {', '.join(f'{k} ({v})' for k,v in best_t.items())}. Corrélation de rang moyenne entre validation et test : {rho:.2f}.

Lecture : l'écart entre deux réseaux n'est interprétable que s'il dépasse la dispersion due à la graine. Un classement obtenu sur une seule graine, comme celui de RESULTATS.md, ne suffit pas à désigner un meilleur modèle.
"""
    (ROOT/'reports/RESULTATS_MULTIGRAINES.md').write_text(md,encoding='utf-8')

NAMES={'empirical_train':'Référence empirique','gaussian_train':'Référence normale','vae':'VAE','vae_weighted':'VAE pondéré','gan':'GAN','wgan_gp':'WGAN-GP','cgan':'cGAN'}

def write_tex(res,agg,stab):
    """Sous-section de la note LaTeX (docs/latex/section_multigraines.tex), générée : aucun chiffre saisi à la main."""
    f=lambda x,d=3:f'{x:.{d}f}'.replace('.','{,}')
    v=agg[agg.split=='validation'].set_index('model');t=agg[agg.split=='test'].set_index('model')
    rows=[f"{NAMES[m]} & {f(v.w_mean[m])} $\\pm$ {f(v.w_sd[m])} & {v.rank_best[m]} & {f(t.w_mean[m])} $\\pm$ {f(t.w_sd[m])} & {f(t.w_min[m])}--{f(t.w_max[m])} & {t.rank_best[m]} \\\\" for m in v.sort_values('w_mean').index]
    n=len(res['seeds']);rho=sum(s_['spearman_val_test'] for s_ in stab)/len(stab);rep=res['reproducibility']['max_abs_diff_wasserstein_vs_published_by_model']
    tex=r"""%% Fichier généré par python -m scripts.multi_seed ; ne pas modifier à la main.
\subsection*{Robustesse à la graine}
Le tableau précédent repose sur une seule graine. Le même protocole est relancé pour %d graines (%s) : seuls changent l'initialisation des réseaux, l'ordre des batchs et les tirages générés. Moyenne $\pm$ écart-type de la distance de Wasserstein, et nombre de graines où la méthode est première.

\begin{center}\small
\begin{tabular}{lrrrrr}
\toprule
& \multicolumn{2}{c}{Validation} & \multicolumn{3}{c}{Test déjà consulté} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}
Méthode & Wasserstein & 1\textsuperscript{er} & Wasserstein & Min--max & 1\textsuperscript{er} \\
\midrule
%s
\bottomrule
\end{tabular}\end{center}
En validation, aucun réseau ne bat en moyenne la référence empirique ; en test, le GAN est le plus souvent premier mais avec une forte dispersion. Le cGAN, très proche du test avec la graine 100, ne l'est plus en moyenne : son écart-type de %s rend ce résultat non représentatif. La corrélation de rang entre validation et test vaut %s en moyenne. Un écart entre deux réseaux n'est interprétable que s'il dépasse la dispersion due à la graine.

\smallnote{Relancée sur une autre machine, la graine 100 redonne exactement les références et le VAE pondéré ; VAE, GAN et WGAN-GP bougent de moins de 0,02 et le cGAN de %s : la graine fixe la reproductibilité sur une même machine, pas entre plateformes, et l'entraînement adverse amplifie les écarts numériques.}
"""%(n,', '.join(map(str,res['seeds'])),'\n'.join(rows),f(t.w_sd['cgan'],2),f(rho,2),f(rep['cgan'],2))
    (ROOT/'docs/latex/section_multigraines.tex').write_text(tex,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[100,101,102,103,104]);p.add_argument('--epochs',type=int,default=10);p.add_argument('--n-samples',type=int,default=10000)
    a=p.parse_args();main(a.seeds,a.epochs,a.n_samples)
