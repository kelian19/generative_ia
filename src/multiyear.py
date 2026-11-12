"""Jeu multi-annuel ERA5 + FIRMS : tableau de cellules-jours, découpage par années, validation sur feux observés."""
import numpy as np
import pandas as pd
from scipy import stats

TRAIN_YEARS = (2001, 2016)
VALIDATION_YEARS = (2017, 2020)
TEST_YEARS = (2021, 2024)


def build_frame(index, ds, keep):
    """Une ligne par maille-jour valide (terre, relief admissible, indice fini) : indice, feux observés, coordonnées."""
    values = index.values
    ok = keep.values[None, :, :] & np.isfinite(values)
    t, i, j = np.nonzero(ok)
    times = pd.DatetimeIndex(index.valid_time.values)[t]
    frame = pd.DataFrame({
        'valid_time': times,
        'latitude': index.latitude.values[i].astype(float),
        'longitude': index.longitude.values[j].astype(float),
        'haines_index': values[t, i, j].astype(float),
        'fire_count': ds.fire_count.values[t, i, j].astype(int),
    })
    frame['year'] = frame.valid_time.dt.year
    return frame


def year_split(frame, train=TRAIN_YEARS, validation=VALIDATION_YEARS, test=TEST_YEARS):
    """Découpage par années entières : aucune date n'est partagée, et les saisons JJAS sont séparées par l'hiver."""
    spans = [train, validation, test]
    for (a, b), (c, d) in zip(spans[:-1], spans[1:]):
        if not a <= b < c <= d:
            raise ValueError('Périodes train < validation < test, disjointes et ordonnées, requises.')
    parts = tuple(frame[(frame.year >= a) & (frame.year <= b)].copy() for a, b in spans)
    if any(p.empty for p in parts):
        raise ValueError('Une période du découpage est vide.')
    return parts


def fire_lift(frame, threshold, bins=None):
    """Probabilité d'au moins un feu par maille-jour selon l'indice, et portée du seuil de déclenchement."""
    fire = (frame.fire_count > 0).to_numpy()
    h = frame.haines_index.to_numpy()
    above = h > threshold
    base = float(fire.mean())
    out = {
        'fire_cell_day_rate': base,
        'threshold': float(threshold),
        'exceedance_rate': float(above.mean()),
        'fire_rate_above': float(fire[above].mean()) if above.any() else float('nan'),
        'fire_rate_below': float(fire[~above].mean()),
        'hit_rate': float(above[fire].mean()) if fire.any() else float('nan'),
        'false_alarm_rate': float((~fire)[above].mean()) if above.any() else float('nan'),
        'lift': float(fire[above].mean() / base) if above.any() and base > 0 else float('nan'),
        'fire_cell_days': int(fire.sum()),
        'cell_days': int(len(frame)),
    }
    if bins is not None:
        cut = pd.cut(frame.haines_index, bins, right=False)
        grouped = pd.DataFrame({'bin': cut, 'fire': fire}).groupby('bin', observed=True).fire.agg(['mean', 'sum', 'size'])
        out['by_bin'] = [{'bin': str(k), 'fire_rate': float(r['mean']), 'fires': int(r['sum']), 'cell_days': int(r['size'])} for k, r in grouped.iterrows()]
    return out


def yearly_association(frame, threshold):
    """Par année : cellules-jours au-dessus du seuil et nombre de détections de feu ; corrélation de rang."""
    g = frame.assign(above=frame.haines_index > threshold, fire=frame.fire_count).groupby('year')
    table = pd.DataFrame({'exceedance_cell_days': g.above.sum(), 'fire_detections': g.fire.sum(), 'cell_days': g.size()})
    if len(table) >= 4:
        rho, p = stats.spearmanr(table.exceedance_cell_days, table.fire_detections)
    else:
        rho, p = float('nan'), float('nan')
    return table.reset_index(), float(rho), float(p)
