"""Construit le jeu multi-annuel ERA5 + FIRMS (France élargie, juin–septembre, 12 UTC).

    python -m scripts.build_multiyear_dataset --start 2001 --end 2024

Sources publiques, sans clé :
- ERA5 (ECMWF/Copernicus) via ARCO-ERA5 : T850, T700, humidité spécifique q850,
  masque terre-mer et géopotentiel de surface ; lecture partielle des seules mailles utiles.
- NASA FIRMS MODIS C6.1 : détections de feux de végétation par jour.

Le C-Haines est recalculé avec le point de rosée issu de q850 (pression de vapeur
exacte à 850 hPa), et non plus d'une humidité relative tronquée.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data_loader import continuous_haines_from_q
from src.sources import era5_raw, firms

ROOT = Path(__file__).resolve().parents[1]
BOX = dict(lat_max=51., lat_min=41., lon_min=-5., lon_max=10.)
FIELDS = [('temperature', 850), ('temperature', 700), ('specific_humidity', 850)]
HOUR = 12
MONTHS = (6, 7, 8, 9)


def season_days(start: int, end: int) -> list[dt.date]:
    days = pd.date_range(f'{start}-01-01', f'{end}-12-31', freq='D')
    return [d.date() for d in days if d.month in MONTHS]


def fetch_day(day, rows, cols):
    return [era5_raw.read_box(era5_raw.pressure_level_url(day, v, lev), HOUR, rows, cols) for v, lev in FIELDS]


def main(start: int, end: int, workers: int, output: str):
    rows, cols, lat, lon = era5_raw.grid_indices(**BOX)
    days = season_days(start, end)
    print(f'{len(days)} jours × {len(FIELDS)} champs à lire…', flush=True)
    cache = ROOT / f'data/cache/era5_fields_{start}_{end}.npy'
    if cache.exists():
        fields = np.load(cache)
        print(f'Champs ERA5 relus depuis {cache.name}', flush=True)
    else:
        fields = np.full((len(days), len(FIELDS), len(lat), len(lon)), np.nan, dtype=np.float32)
        with ThreadPoolExecutor(workers) as pool:
            for k, values in enumerate(pool.map(lambda d: fetch_day(d, rows, cols), days)):
                fields[k] = np.stack(values)
                if k % 250 == 0:
                    print(f'  {k}/{len(days)} {days[k]}', flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, fields)
    if not np.isfinite(fields).all():
        raise ValueError('Valeur ERA5 manquante.')
    t850, t700, q850 = fields[:, 0] - 273.15, fields[:, 1] - 273.15, fields[:, 2]
    haines = continuous_haines_from_q(t850, t700, q850, 850.)

    lsm = era5_raw.read_box(era5_raw.static_url('land_sea_mask'), 0, rows, cols)
    elevation = era5_raw.read_box(era5_raw.static_url('geopotential'), 0, rows, cols) / 9.80665

    dates = pd.DatetimeIndex(pd.to_datetime(days))
    fire_frames, firms_sizes = [], {}
    for year in range(start, end + 1):
        raw, sizes = firms.download_year(year)
        firms_sizes[year] = sizes
        fire_frames.append(firms.filter_vegetation(raw))
        print(f'  FIRMS {year}: {len(raw)} détections, {len(fire_frames[-1])} retenues', flush=True)
    fires = firms.grid_counts(pd.concat(fire_frames, ignore_index=True), dates, lat, lon)

    ds = xr.Dataset(
        {
            'haines_index': (('valid_time', 'latitude', 'longitude'), haines.astype(np.float32),
                             {'long_name': 'Continuous Haines index (Mills & McCaw 2010)', 'units': '1',
                              'dewpoint': 'from specific humidity at 850 hPa (exact vapour pressure, Magnus inversion)'}),
            'fire_count': (('valid_time', 'latitude', 'longitude'), np.minimum(fires, 255).astype(np.uint8),
                           {'long_name': 'MODIS C6.1 vegetation fire detections (type 0, confidence >= 30) on that UTC day',
                            'units': 'count'}),
            't850': (('valid_time', 'latitude', 'longitude'), t850.astype(np.float32), {'units': 'degC'}),
            'dewpoint_depression_850': (('valid_time', 'latitude', 'longitude'),
                                        (t850 - _dewpoint(q850)).astype(np.float32), {'units': 'K'}),
            'land_sea_mask': (('latitude', 'longitude'), lsm, {'units': '1', 'source': 'ERA5 static'}),
            'surface_elevation': (('latitude', 'longitude'), elevation.astype(np.float32),
                                  {'units': 'm', 'source': 'ERA5 surface geopotential / g0'}),
        },
        coords={'valid_time': dates + pd.Timedelta(hours=HOUR), 'latitude': lat, 'longitude': lon},
        attrs={'title': 'ERA5 C-Haines and FIRMS MODIS fire detections, JJAS 12 UTC',
               'era5_source': era5_raw.SOURCE_DOC, 'firms_source': firms.SOURCE_DOC,
               'era5_licence': 'Copernicus licence (attribution: Contains modified Copernicus Climate Change Service information)',
               'firms_credit': 'NASA FIRMS, MODIS Collection 6.1 (MCD14ML)', 'box': json.dumps(BOX),
               'built': dt.datetime.now(dt.UTC).isoformat(timespec='seconds')},
    )
    encoding = {name: {'zlib': True, 'complevel': 6} for name in ds.data_vars}
    for name in ('haines_index', 't850', 'dewpoint_depression_850'):
        encoding[name].update(dtype='int16', scale_factor=0.01, _FillValue=-32768)
    buffer = io.BytesIO()
    ds.to_netcdf(buffer, engine='h5netcdf', encoding=encoding)
    path = ROOT / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    manifest = {'file': output, 'sha256': hashlib.sha256(buffer.getvalue()).hexdigest(), 'bytes': len(buffer.getvalue()),
                'days': len(days), 'years': [start, end], 'months': list(MONTHS), 'hour_utc': HOUR, 'box': BOX,
                'era5_fields': [f'{v}@{lev}hPa' for v, lev in FIELDS], 'era5_url_pattern': era5_raw.pressure_level_url(days[0], '<var>', 0),
                'firms_url_pattern': firms.URL, 'firms_countries': list(firms.COUNTRIES), 'firms_bytes': firms_sizes,
                'firms_filter': {'type': 0, 'min_confidence': firms.MIN_CONFIDENCE},
                'fire_detections_kept_in_window': int(fires.sum())}
    (ROOT / 'reports/data_manifest_multiyear.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('file', 'sha256', 'bytes', 'days', 'fire_detections_kept_in_window')}, indent=2))


def _dewpoint(q):
    from src.data_loader import dewpoint_from_q
    return dewpoint_from_q(q, 850.)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=2001)
    p.add_argument('--end', type=int, default=2024)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--output', default='data/era5_firms_jjas_2001_2024.nc')
    a = p.parse_args()
    main(a.start, a.end, a.workers, a.output)
