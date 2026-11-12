"""Remplit la section LaTeX et le rapport Markdown du benchmark multi-annuel depuis reports/multiyear_benchmark.json.

    python -m scripts.make_multiyear_section

Aucun chiffre n'est saisi à la main, sauf la part de détections retenues par le filtre FIRMS (journal de
construction du jeu, non stockée dans le manifeste) : constantes KEEP_* ci-dessous.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BS = chr(92)
KEEP_TRAIN, KEEP_TEST = 77, 63  # % de détections FIRMS retenues (type 0, confiance >= 30), journal de construction
NAMES = {'empirical_train': 'Référence empirique', 'gaussian_train': 'Référence normale', 'vae': 'VAE', 'vae_weighted': 'VAE pondéré', 'gan': 'GAN',
         'wgan_gp': 'WGAN-GP', 'cgan': 'cGAN'}


def num(x, digits=2):
    return f'{x:.{digits}f}'.replace('.', ',')


def big(n):
    return f'{int(n):,}'.replace(',', BS + ',')


def main():
    r = json.loads((ROOT / 'reports/multiyear_benchmark.json').read_text(encoding='utf-8'))
    m, fv = r['models'], r['fire_validation']
    test0 = m['empirical_train']['test']
    best = min((k for k in m if not m[k].get('baseline')), key=lambda k: m[k]['test']['wasserstein'])
    rows = []
    for k in NAMES:
        t, v = m[k]['test'], m[k]['validation']
        rows.append(f"{NAMES[k]} & {num(v['wasserstein'], 3)} & {num(t['wasserstein'], 3)} & {num(t['q99_gen'])} & {num(t['trigger_probability'] * 100)}"
                    + BS + ',' + BS + '% ' + BS + BS)
    keys = {
        'N_CELLS': big(r['index_definition']['cells_kept']), 'ROWS_ALL': big(r['rows']['train'] + r['rows']['validation'] + r['rows']['test']),
        'ROWS_TRAIN': big(r['rows']['train']), 'ROWS_VAL': big(r['rows']['validation']), 'ROWS_TEST': big(r['rows']['test']),
        'FIRE_TRAIN': big(fv['train']['fire_cell_days']), 'FIRE_VAL': big(fv['validation']['fire_cell_days']), 'FIRE_TEST': big(fv['test']['fire_cell_days']),
        'FIT_ROWS': big(r['rows']['train_fit_sample']), 'TRIGGER': num(r['trigger']), 'Q99_REAL': num(test0['q99_real']),
        'OBS_TRIG': num(test0['observed_trigger_probability'] * 100), 'MODEL_ROWS': '\n'.join(rows),
        'BEST_LABEL': NAMES[best], 'BEST_W': num(m[best]['test']['wasserstein'], 3), 'EMP_W': num(test0['wasserstein'], 3),
        'GAN_W': num(m['gan']['test']['wasserstein'], 3), 'WGAN_Q99': num(m['wgan_gp']['test']['q99_gen']),
        'RATE_ABOVE': num(fv['test']['fire_rate_above'] * 100), 'RATE_BELOW': num(fv['test']['fire_rate_below'] * 100),
        'LIFT_TEST': num(fv['test']['lift'], 1), 'LIFT_TRAIN': num(fv['train']['lift'], 1), 'LIFT_VAL': num(fv['validation']['lift'], 1),
        'HIT_TEST': num(fv['test']['hit_rate'] * 100, 0), 'FALSE_ALARM': num(fv['test']['false_alarm_rate'] * 100, 1),
        'RHO': num(r['yearly_association']['spearman_rho']), 'PVAL': num(r['yearly_association']['p_value'], 3),
        'KEEP_TRAIN': str(KEEP_TRAIN), 'KEEP_TEST': str(KEEP_TEST),
        'FR_TRAIN': num(fv['train']['fire_cell_day_rate'] * 100), 'FR_TEST': num(fv['test']['fire_cell_day_rate'] * 100),
    }
    # Garde-fous : le texte interprétatif suppose ces faits ; la génération échoue s'ils ne tiennent plus.
    non_baseline = [k for k in m if not m[k].get('baseline')]
    assert best == 'vae_weighted', 'Le meilleur modèle a changé : relire le texte.'
    assert max(non_baseline, key=lambda k: m[k]['test']['wasserstein']) == 'gan', 'Le GAN n’est plus le moins bon : relire le texte.'
    assert all(m[k]['test']['trigger_probability'] < test0['observed_trigger_probability'] for k in m), 'Sous-estimation générale remise en cause.'
    assert min(non_baseline, key=lambda k: abs(m[k]['test']['q99_gen'] - test0['q99_real'])) == 'wgan_gp', 'Q99 le plus proche : relire le texte.'
    template = (ROOT / 'docs/latex/section_multiannuelle.tex.tmpl').read_text(encoding='utf-8')
    for key, value in keys.items():
        template = template.replace(f'@@{key}@@', value)
    assert '@@' not in template, 'Marqueur non rempli.'
    (ROOT / 'docs/latex/section_multiannuelle.tex').write_text(template, encoding='utf-8')
    md = [
        '# Résultats multi-annuels (ERA5 + FIRMS, JJAS 2001–2024)', '',
        f"Fichier `{r['data_file']}` (SHA256 `{r['data_sha256']}`). Apprentissage 2001–2016, validation 2017–2020, test 2021–2024 ; "
        f"seuil de démonstration {num(r['trigger'])} (quantile 95 % de l'apprentissage). Graine {r['seed']}, {r['epochs_budget_per_model']} époques, "
        f"sous-échantillon d'apprentissage de {r['rows']['train_fit_sample']:,}".replace(',', ' ') + ' mailles-jours.', '',
        '| Méthode | Wasserstein val. | Wasserstein test | Q99 généré (test) | P(indice > seuil) test |', '|---|---:|---:|---:|---:|']
    for k in NAMES:
        t, v = m[k]['test'], m[k]['validation']
        md.append(f"| {NAMES[k]} | {num(v['wasserstein'], 3)} | {num(t['wasserstein'], 3)} | {num(t['q99_gen'])} | {num(t['trigger_probability'] * 100)} % |")
    md += ['', f"Q99 observé sur le test : {num(test0['q99_real'])} ; fréquence observée de dépassement : {num(test0['observed_trigger_probability'] * 100)} %.", '',
           '## Lien avec les feux observés (FIRMS MODIS, type 0, confiance ≥ 30 %)', '',
           '| Période | Taux de feu (maille-jour) | Au-dessus du seuil | Sous le seuil | Surcote | Part des feux capturée |', '|---|---:|---:|---:|---:|---:|']
    for s, label in [('train', 'Apprentissage'), ('validation', 'Validation'), ('test', 'Test')]:
        f = fv[s]
        md.append(f"| {label} | {num(f['fire_cell_day_rate'] * 100)} % | {num(f['fire_rate_above'] * 100)} % | {num(f['fire_rate_below'] * 100)} % | ×{num(f['lift'], 1)} | {num(f['hit_rate'] * 100, 0)} % |")
    md += ['', f"Corrélation de rang saisonnière (24 saisons) entre maille-jours au-dessus du seuil et détections : {num(r['yearly_association']['spearman_rho'])} "
           f"(p = {num(r['yearly_association']['p_value'], 3)}, indicative : mailles dépendantes).", '', '## Limites', '']
    md += [f'- {x}.' for x in r['limitations']]
    md += [f'- La part de détections retenues par le filtre passe d’environ {KEEP_TRAIN} % (2001–2016) à {KEEP_TEST} % (2021–2024), d’après le journal de construction ; non analysée.',
           '- Une seule graine : les écarts entre modèles ne constituent pas une supériorité établie. Le classement diffère de celui de juillet 2023 (le GAN y était le meilleur).']
    (ROOT / 'reports/RESULTATS_MULTIANNUEL.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('section et rapport écrits ; meilleur modèle :', best)


if __name__ == '__main__':
    main()
