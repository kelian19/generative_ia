"""Benchmark multi-annuel (JJAS 2001–2024) avec validation sur détections de feux observées.

    python -m scripts.build_multiyear_dataset          # une fois (~30 min, réseau)
    python -m src.run_multiyear --epochs 10 --seed 100

Protocole : apprentissage 2001–2016, validation 2017–2020, test 2021–2024 (années entières). Le test n'est
consulté qu'une fois, après sélection du seuil et des hyperparamètres sur apprentissage/validation (déjà fixés
par le benchmark de juillet 2023). Les réseaux sont entraînés sur un sous-échantillon aléatoire fixe de
l'apprentissage ; les métriques sont calculées sur toutes les mailles-jours des périodes d'évaluation.
"""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
import importlib.metadata as md

import numpy as np
import pandas as pd
import torch

import config
from src.actuarial import compute_prime_pure
from src.data_loader import MULTIYEAR_PATH, load_multiyear
from src.evaluation import compute_metrics, generate_cgan_samples, generate_gan_samples, generate_vae_samples
from src.multiyear import TEST_YEARS, TRAIN_YEARS, VALIDATION_YEARS, build_frame, fire_lift, year_split, yearly_association
from src.preprocessing import fit_transform, make_loader, transform
from src.training.train_cgan import prepare_cgan_data, train_cgan
from src.training.train_gan import train_gan
from src.training.train_vae import train_vae, train_vae_pro
from src.training.train_wgan import train_wgan

ROOT = Path(__file__).resolve().parents[1]
QUANTILE_GRID = np.round(np.linspace(.005, .995, 100), 4)
TAIL_LEVELS = (.9, .95, .99, .999)
BINS = [-np.inf, 0, 2, 4, 6, 8, 10, 12, np.inf]


def _evaluate(samples, part, trigger, tail_rows):
    real = part.haines_index.to_numpy()
    result = compute_metrics(real, samples)
    result['tail_quantiles'] = {str(p): {'real': float(np.quantile(real, p)), 'generated': float(np.quantile(samples, p))} for p in TAIL_LEVELS}
    probability, payment = compute_prime_pure(samples, trigger, config.INDEMNITE)
    observed = float((real > trigger).mean())
    result.update(trigger_probability=probability, expected_payment_per_observation=payment, observed_trigger_probability=observed,
                  trigger_probability_ratio=float(probability / observed) if observed > 0 else float('nan'))
    result['quantile_curve'] = {'p': QUANTILE_GRID.tolist(), 'generated': np.quantile(samples, QUANTILE_GRID).tolist()}
    return result


def run(epochs=10, n_train=200_000, n_samples=100_000, seed=100, data=MULTIYEAR_PATH, output='reports'):
    if epochs < 2 or n_train < 1000 or n_samples < 1000:
        raise ValueError('Au moins 2 époques, 1 000 observations d’apprentissage et 1 000 simulations requises.')
    config.set_seed(seed)
    torch.set_num_threads(2)
    path = Path(data) if Path(data).is_absolute() else ROOT / data
    ds, index, keep = load_multiyear(str(path))
    frame = build_frame(index, ds, keep)
    train, val, test = year_split(frame)
    trigger = float(np.quantile(train.haines_index, .95))
    lift = {name: fire_lift(part, trigger, BINS) for name, part in [('train', train), ('validation', val), ('test', test)]}
    yearly, rho, p_value = yearly_association(frame, trigger)
    full_train = train
    train, lmbda, scaler, shift = fit_transform(train)
    val, test = transform(val, scaler), transform(test, scaler)
    rng = np.random.default_rng(seed)
    fit_rows = train.iloc[np.sort(rng.choice(len(train), min(n_train, len(train)), replace=False))]
    report = {
        'data_file': str(path.name), 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'index_definition': index.attrs,
        'periods': {'train': TRAIN_YEARS, 'validation': VALIDATION_YEARS, 'test': TEST_YEARS},
        'rows': {'train': len(full_train), 'train_fit_sample': len(fit_rows), 'validation': len(val), 'test': len(test)},
        'trigger': trigger, 'trigger_origin': 'quantile 95 % de l’apprentissage complet ; seuil de démonstration, non contractuel',
        'indemnity_demo': config.INDEMNITE, 'seed': seed, 'epochs_budget_per_model': epochs, 'n_samples': n_samples,
        'fire_validation': lift, 'yearly_association': {'spearman_rho': rho, 'p_value': p_value, 'years': yearly.to_dict('records'),
                                                         'note': '24 années, dépendance spatiale : p-value indicative'},
        'models': {}, 'versions': {name: md.version(name) for name in ['torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'xarray', 'h5netcdf']},
        'python': platform.python_version(),
        'limitations': ['Détections MODIS de feux de végétation (type 0, confiance ≥ 30 %) : inclut des brûlages agricoles, pas des pertes assurées',
                        'Indice à 12 UTC ; les feux sont comptés sur la journée UTC entière',
                        'Mailles dépendantes spatialement et temporellement : p-values KS et corrélations de rang indicatives',
                        'Distributions scalaires par maille-jour, pas des champs cohérents',
                        'Le seuil 95 % est un seuil de démonstration, pas un déclencheur contractuel optimisé'],
        'real_quantile_curve_validation': {'p': QUANTILE_GRID.tolist(), 'real': np.quantile(val.haines_index, QUANTILE_GRID).tolist()},
        'real_quantile_curve_test': {'p': QUANTILE_GRID.tolist(), 'real': np.quantile(test.haines_index, QUANTILE_GRID).tolist()},
    }
    started = time.perf_counter()
    gen = np.random.default_rng(seed)
    baselines = {'empirical_train': gen.choice(full_train.haines_index.to_numpy(), n_samples, replace=True),
                 'gaussian_train': gen.normal(full_train.haines_index.mean(), full_train.haines_index.std(ddof=0), n_samples)}
    for name, samples in baselines.items():
        report['models'][name] = {'validation': _evaluate(samples, val, trigger, 0), 'test': _evaluate(samples, test, trigger, 0), 'baseline': True}
    val_cond_rows = val.sample(min(50_000, len(val)), random_state=seed)
    test_cond_rows = test.sample(min(50_000, len(test)), random_state=seed)
    for name, cfg0, trainer in [('vae', config.VAE, train_vae), ('vae_weighted', config.VAE_PRO, train_vae_pro), ('gan', config.GAN, train_gan),
                                ('wgan_gp', config.WGAN, train_wgan), ('cgan', config.CGAN, train_cgan)]:
        config.set_seed(seed)
        cfg = copy.deepcopy(cfg0)
        cfg.update(epochs=epochs, device='cpu')
        if name == 'vae_weighted':
            cfg.update(epochs=epochs // 2, weighted_epochs=epochs - epochs // 2)
        then = time.perf_counter()
        if name == 'cgan':
            loader, conditions, encoder = prepare_cgan_data(fit_rows, batch_size=config.BATCH_SIZE, seed=seed)
            model, _ = trainer(loader, conditions, cfg)
            tv = torch.as_tensor(encoder.transform(val_cond_rows), dtype=torch.float32)
            tt = torch.as_tensor(encoder.transform(test_cond_rows), dtype=torch.float32)
            sv = generate_cgan_samples(model, cfg['latent_dim'], tv, scaler, lmbda, shift, n_samples, seed)
            st = generate_cgan_samples(model, cfg['latent_dim'], tt, scaler, lmbda, shift, n_samples, seed)
        else:
            loader, _ = make_loader(fit_rows, batch_size=config.BATCH_SIZE, seed=seed)
            model, _ = trainer(loader, cfg)
            generate = generate_vae_samples if name.startswith('vae') else generate_gan_samples
            sv = st = generate(model, cfg['latent_dim'], scaler, lmbda, shift, n_samples, seed)
        report['models'][name] = {'validation': _evaluate(sv, val, trigger, 0), 'test': _evaluate(st, test, trigger, 0), 'config': cfg,
                                  'seconds': time.perf_counter() - then, 'history': model.training_history_}
        print(name, round(report['models'][name]['test']['wasserstein'], 4), flush=True)
    report['seconds'] = time.perf_counter() - started
    folder = Path(output) if Path(output).is_absolute() else ROOT / output
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'multiyear_benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False, default=float),encoding='utf-8')
    yearly.to_csv(folder / 'multiyear_yearly.csv', index=False)
    rows = [{'model': m, 'split': s, **{k: v for k, v in values[s].items() if not isinstance(v, (dict, list))}}
            for m, values in report['models'].items() for s in ('validation', 'test')]
    pd.DataFrame(rows).to_csv(folder / 'multiyear_benchmark.csv', index=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n-train', type=int, default=200_000)
    parser.add_argument('--n-samples', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--data', default=MULTIYEAR_PATH)
    parser.add_argument('--output', default='reports')
    a = parser.parse_args()
    run(a.epochs, a.n_train, a.n_samples, a.seed, a.data, a.output)
