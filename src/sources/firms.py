"""Détections de feux actifs NASA FIRMS (MODIS C6.1), fichiers annuels publics par pays.

Les fichiers ``modis_<année>_<pays>.csv`` sont téléchargeables sans clé. Seules les
détections ``type == 0`` (feu de végétation présumé) et de confiance ≥ 30 % sont
retenues ; volcans, sources fixes (industrie, torchères) et détections offshore sont exclus.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

URL = 'https://firms.modaps.eosdis.nasa.gov/data/country/modis/{year}/modis_{year}_{country}.csv'
SOURCE_DOC = 'https://firms.modaps.eosdis.nasa.gov/country/'
# Pays dont le territoire intersecte la boîte 41–51°N, 5°W–10°E.
COUNTRIES = ('France', 'Spain', 'Italy', 'Switzerland', 'Germany', 'Belgium', 'Luxembourg', 'United_Kingdom', 'Andorra')
TINY_COUNTRIES = ('Andorra', 'Luxembourg', 'Switzerland')
MIN_CONFIDENCE = 30


def download_year(year: int, countries=COUNTRIES) -> tuple[pd.DataFrame, dict]:
    frames, sizes = [], {}
    for country in countries:
        url = URL.format(year=year, country=country)
        try:
            payload = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'generative-ia-research'}), timeout=120).read()
        except urllib.error.HTTPError as exc:
            # Les très petits pays n'ont pas de fichier les années sans détection : absence consignée, pas silencieuse.
            if exc.code == 404 and country in TINY_COUNTRIES:
                sizes[country] = None
                continue
            raise RuntimeError(f'Fichier FIRMS indisponible : {url} ({exc.code}).') from exc
        sizes[country] = len(payload)
        frame = pd.read_csv(io.BytesIO(payload), usecols=['latitude', 'longitude', 'acq_date', 'confidence', 'type'])
        frame['country'] = country
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), sizes


def filter_vegetation(df: pd.DataFrame) -> pd.DataFrame:
    keep = (df['type'] == 0) & (df['confidence'] >= MIN_CONFIDENCE)
    return df.loc[keep].copy()


def grid_counts(df: pd.DataFrame, dates: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray, res: float = 0.25) -> np.ndarray:
    """Nombre de détections par jour et maille ERA5 la plus proche ; dates absentes ignorées."""
    counts = np.zeros((len(dates), len(lat), len(lon)), dtype=np.uint16)
    if df.empty:
        return counts
    t = dates.get_indexer(pd.to_datetime(df['acq_date']))
    i = np.rint((lat[0] - df['latitude'].to_numpy()) / res).astype(int)
    j = np.rint((df['longitude'].to_numpy() - lon[0]) / res).astype(int)
    ok = (t >= 0) & (i >= 0) & (i < len(lat)) & (j >= 0) & (j < len(lon))
    np.add.at(counts, (t[ok], i[ok], j[ok]), 1)
    return counts
