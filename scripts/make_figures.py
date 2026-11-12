"""Figures vectorielles de la note LaTeX, tirées des rapports et du jeu multi-annuel.

    python -m scripts.make_figures

Sorties : docs/latex/figures/*.pdf. Les figures du benchmark de juillet 2023 lisent reports/*.json ;
celles du jeu multi-annuel exigent reports/multiyear_benchmark.json et data/era5_firms_jjas_2001_2024.nc.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/latex/figures'
FOREST, AMBER, MUTED, INK = '#1F4E79', '#B5651D', '#65756A', '#243C35'
LIGHT, MID, RED = '#9DB8D2', '#5B87B0', '#B03A2E'
MODEL_ORDER = ['empirical_train', 'gaussian_train', 'vae', 'vae_weighted', 'gan', 'wgan_gp', 'cgan']
LABELS = {'empirical_train': 'empirique\n(train)', 'gaussian_train': 'normale\n(train)', 'vae': 'VAE', 'vae_weighted': 'VAE\npondéré',
          'gan': 'GAN', 'wgan_gp': 'WGAN-\nGP', 'cgan': 'cGAN'}
COLORS = {'empirical_train': MUTED, 'gaussian_train': '#A9B4AD', 'vae': FOREST, 'vae_weighted': MID, 'gan': AMBER, 'wgan_gp': RED, 'cgan': '#7D5BA6'}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#DDE3EA', 'grid.linewidth': .5,
    'axes.axisbelow': True, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'legend.frameon': False,
})


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def comma(x, digits=2):
    return f'{x:.{digits}f}'.replace('.', ',')


def title(ax, text):
    ax.set_title(text, loc='left', fontsize=9, fontweight='bold', color=FOREST)


def benchmark_panels(report, observed_q99_key='q99_real', figsize=(7.1, 2.7), unit_note=''):
    """Wasserstein validation/test, Q99 généré, probabilité de déclenchement : trois lectures du même benchmark."""
    models = [m for m in MODEL_ORDER if m in report['models']]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1.25, 1, 1]})
    a, b, c = axes
    w = .38
    val = [report['models'][m]['validation']['wasserstein'] for m in models]
    test = [report['models'][m]['test']['wasserstein'] for m in models]
    a.bar(x - w / 2, val, w, color=LIGHT, zorder=3, label='validation')
    a.bar(x + w / 2, test, w, color=FOREST, zorder=3, label='test')
    a.set_xticks(x)
    a.set_xticklabels([LABELS[m] for m in models], fontsize=6.5)
    a.set_ylabel('distance de Wasserstein')
    a.set_ylim(0, max(val + test) * 1.28)
    a.legend(fontsize=7, loc='upper left', ncol=2)
    title(a, 'Distance marginale')
    q99 = [report['models'][m]['test']['q99_gen'] for m in models]
    real = report['models'][models[0]]['test'][observed_q99_key]
    b.bar(x, q99, .62, color=[COLORS[m] for m in models], zorder=3)
    b.axhline(real, color=INK, lw=1.2, zorder=4)
    b.text(-.45, real + .1, f'observé : {comma(real)}', ha='left', va='bottom', fontsize=7, bbox=dict(fc='white', ec='none', pad=1, alpha=.85), zorder=6)
    b.set_xticks(x)
    b.set_xticklabels([LABELS[m] for m in models], fontsize=6.5)
    b.set_ylim(min(q99 + [real]) - 1.5, max(q99 + [real]) + .8)
    b.set_ylabel('quantile 99 % généré')
    title(b, 'Queue (test)')
    prob = [report['models'][m]['test']['trigger_probability'] * 100 for m in models]
    obs = report['models'][models[0]]['test']['observed_trigger_probability'] * 100
    c.bar(x, prob, .62, color=[COLORS[m] for m in models], zorder=3)
    c.axhline(obs, color=INK, lw=1.2, zorder=4)
    c.text(-.45, obs + .05 * max(prob), f'observé : {comma(obs)} %', ha='left', va='bottom', fontsize=7, bbox=dict(fc='white', ec='none', pad=1, alpha=.85), zorder=6)
    c.set_ylim(0, max(prob + [obs]) * 1.3)
    c.set_xticks(x)
    c.set_xticklabels([LABELS[m] for m in models], fontsize=6.5)
    c.set_ylabel('P(indice > seuil) (%)')
    title(c, 'Déclenchement (test)')
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    fig.tight_layout(w_pad=1.6)
    return fig


def fig_wgan_stability():
    final, initial = load('reports/benchmark.json'), load('reports/benchmark_initial.json')
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={'width_ratios': [1.15, 1]})
    for report, label, color, style in [(initial, 'WGAN initial (divergent)', RED, '--'), (final, 'WGAN corrigé', FOREST, '-')]:
        history = report['models']['wgan_gp']['history']
        a.plot([h['epoch'] for h in history], [h['critic_loss'] for h in history], color=color, ls=style, lw=1.8, marker='o', ms=3, label=label)
    a.set_xlabel('époque')
    a.set_ylabel('perte du critique')
    a.legend(fontsize=7)
    title(a, 'Perte du critique par époque')
    labels = ['WGAN\ninitial', 'WGAN\ncorrigé', 'GAN', 'référence\nempirique']
    values = [initial['models']['wgan_gp']['test']['wasserstein'], final['models']['wgan_gp']['test']['wasserstein'],
              final['models']['gan']['test']['wasserstein'], final['models']['empirical_train']['test']['wasserstein']]
    b.bar(range(4), values, .6, color=[RED, FOREST, AMBER, MUTED], zorder=3)
    for i, v in enumerate(values):
        b.text(i, v * 1.25, comma(v), ha='center', fontsize=7)
    b.set_yscale('log')
    b.set_ylim(.2, 120)
    b.set_xticks(range(4))
    b.set_xticklabels(labels, fontsize=7)
    b.set_ylabel('Wasserstein, test (échelle log)')
    title(b, 'Effet de la correction')
    fig.tight_layout(w_pad=2)
    return fig


def fig_july_benchmark():
    report = load('reports/benchmark.json')
    return benchmark_panels(report)


def fig_maps(ds, keep, years):
    """Indice moyen et densité de feux observés sur la zone d'étude (mailles terrestres retenues)."""
    lat, lon = ds.latitude.values, ds.longitude.values
    mean_h = np.where(keep.values, ds.haines_index.values.mean(axis=0), np.nan)
    fires = np.where(keep.values, ds.fire_count.values.sum(axis=0) / years, np.nan)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    ext = [lon[0] - .125, lon[-1] + .125, lat[-1] - .125, lat[0] + .125]
    im = a.imshow(mean_h, extent=ext, cmap='YlOrBr', aspect='auto', vmin=np.nanpercentile(mean_h, 2), vmax=np.nanpercentile(mean_h, 98))
    fig.colorbar(im, ax=a, shrink=.85, pad=.02).set_label('C-Haines moyen, JJAS 12 UTC', fontsize=7.5)
    im2 = b.imshow(fires, extent=ext, cmap='Reds', aspect='auto', vmin=0, vmax=np.nanpercentile(fires, 99))
    fig.colorbar(im2, ax=b, shrink=.85, pad=.02).set_label('détections de feux par an (JJAS)', fontsize=7.5)
    for ax, text in [(a, 'Indice moyen'), (b, 'Feux observés (MODIS)')]:
        ax.set_xlabel('longitude (°E)')
        ax.grid(False)
        title(ax, text)
        ax.set_facecolor('#E8ECEF')
    a.set_ylabel('latitude (°N)')
    fig.tight_layout(w_pad=1.5)
    return fig


def fig_fire_lift(report):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={'width_ratios': [1.3, 1]})
    trig = report['trigger']
    for split, color, off in [('train', LIGHT, -.2), ('test', FOREST, .2)]:
        rows = report['fire_validation'][split]['by_bin']
        a.bar(np.arange(len(rows)) + off, [r['fire_rate'] * 100 for r in rows], .38, color=color, zorder=3,
              label={'train': 'apprentissage (2001–2016)', 'test': 'test (2021–2024)'}[split])
    labels = ['< 0', '0–2', '2–4', '4–6', '6–8', '8–10', '10–12', '≥ 12']
    assert len(labels) == len(rows)
    a.set_xticks(range(len(rows)))
    a.set_xticklabels(labels, fontsize=7, rotation=30)
    a.set_xlabel('classe de C-Haines')
    a.set_ylabel('maille-jours avec au moins un feu (%)')
    a.legend(fontsize=7, loc='upper left')
    title(a, 'Feux observés selon l’indice')
    t = report['fire_validation']['test']
    metrics = [('taux de feu\nsous le seuil', t['fire_rate_below'] * 100), ('taux de feu\nau-dessus', t['fire_rate_above'] * 100)]
    b.bar([0, 1], [m[1] for m in metrics], .55, color=[LIGHT, FOREST], zorder=3)
    for i, (_, v) in enumerate(metrics):
        b.text(i, v + .02 * max(m[1] for m in metrics), comma(v, 2) + ' %', ha='center', fontsize=7.5)
    b.set_xticks([0, 1])
    b.set_xticklabels([m[0] for m in metrics], fontsize=7.5)
    b.set_ylabel('% de maille-jours avec feu (test)')
    b.text(.5, .62, f"seuil = {comma(trig)}\nsurcote ×{comma(t['lift'], 1)}\ncapture {comma(t['hit_rate'] * 100, 0)} % des feux", transform=b.transAxes, ha='center', fontsize=7.5,
           bbox=dict(boxstyle='round,pad=.4', fc='#EEF3F8', ec='none'))
    title(b, 'Seuil de déclenchement (test)')
    fig.tight_layout(w_pad=2)
    return fig


def fig_yearly(report):
    rows = report['yearly_association']['years']
    x = np.array([r['exceedance_cell_days'] for r in rows], float)
    y = np.array([r['fire_detections'] for r in rows], float)
    years = [r['year'] for r in rows]
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    colors = [FOREST if yr >= 2021 else (MID if yr >= 2017 else LIGHT) for yr in years]
    ax.scatter(x, y, s=34, c=colors, zorder=3, edgecolor='white', lw=.6)
    for xi, yi, yr in zip(x, y, years):
        ax.annotate(str(yr), (xi, yi), xytext=(3, 3), textcoords='offset points', fontsize=6.5)
    ax.set_xlabel('maille-jours au-dessus du seuil (par saison JJAS)')
    ax.set_ylabel('détections de feux de végétation (par saison)')
    ra = report['yearly_association']
    ax.text(.02, .96, f"rang de Spearman : {comma(ra['spearman_rho'])}\n(p = {comma(ra['p_value'], 3)}, indicative : dépendance spatiale)", transform=ax.transAxes, va='top', fontsize=7.5)
    for c, l in [(LIGHT, 'apprentissage'), (MID, 'validation'), (FOREST, 'test')]:
        ax.scatter([], [], c=c, s=30, label=l)
    ax.legend(loc='lower right', fontsize=7)
    fig.tight_layout()
    return fig


def fig_quantiles(report):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.1), gridspec_kw={'width_ratios': [1, 1]})
    p = np.array(report['real_quantile_curve_test']['p'])
    real = np.array(report['real_quantile_curve_test']['real'])
    for ax, mask, text in [(a, p > 0, 'Toute la distribution'), (b, p >= .9, 'Queue supérieure (p ≥ 0,9)')]:
        lo, hi = real[mask].min(), real[mask].max()
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, zorder=2)
        for m in MODEL_ORDER:
            if m not in report['models']:
                continue
            q = np.array(report['models'][m]['test']['quantile_curve']['generated'])
            ax.plot(real[mask], q[mask], color=COLORS[m], lw=1.5, label=LABELS[m].replace('\n', ' '), zorder=3)
        ax.set_xlabel('quantiles observés (test)')
        title(ax, text)
    a.set_ylabel('quantiles générés')
    b.legend(fontsize=6.5, loc='upper left', ncol=1)
    fig.tight_layout(w_pad=2)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {'fig_july_benchmark': fig_july_benchmark(), 'fig_wgan_stability': fig_wgan_stability()}
    multi = ROOT / 'reports/multiyear_benchmark.json'
    if multi.exists():
        from src.data_loader import load_multiyear
        report = load('reports/multiyear_benchmark.json')
        ds, _, keep = load_multiyear(str(ROOT / 'data' / report['data_file']))
        years = len(set(ds.valid_time.dt.year.values.tolist()))
        figures.update({'fig_maps': fig_maps(ds, keep, years), 'fig_multi_benchmark': benchmark_panels(report), 'fig_fire_lift': fig_fire_lift(report),
                        'fig_yearly': fig_yearly(report), 'fig_quantiles': fig_quantiles(report)})
    for name, fig in figures.items():
        fig.savefig(OUT / f'{name}.pdf')
        plt.close(fig)
        print('écrit', name)


if __name__ == '__main__':
    main()
