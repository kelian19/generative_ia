"""Lecture partielle des fichiers ERA5 journaliers publics (ARCO-ERA5, Google Cloud).

Chaque fichier ``raw/date-variable-pressure_level/AAAA/MM/JJ/<variable>/<niveau>.nc``
est un NetCDF classique (CDF-1/CDF-2) non compressé : 24 enregistrements horaires,
grille 0,25° globale (721 × 1440), valeurs int16 avec ``scale_factor``/``add_offset``.
Le format classique permet de lire seulement les lignes de latitude utiles par une
requête HTTP ``Range`` (≈120 ko au lieu de 50 Mo par fichier).
"""
from __future__ import annotations

import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

BASE_URL = 'https://storage.googleapis.com/gcp-public-data-arco-era5/raw'
SOURCE_DOC = 'https://github.com/google-research/arco-era5'
NLAT, NLON, RES = 721, 1440, 0.25
_TYPES = {1: ('b', 1), 2: ('c', 1), 3: ('h', 2), 4: ('i', 4), 5: ('f', 4), 6: ('d', 8)}


@dataclass(frozen=True)
class Variable:
    name: str
    dims: tuple[int, ...]
    attrs: dict
    nc_type: int
    vsize: int
    begin: int


@dataclass(frozen=True)
class Header:
    version: int
    numrecs: int
    dims: list[tuple[str, int]]
    variables: dict[str, Variable]

    @property
    def record_size(self) -> int:
        record = [v for v in self.variables.values() if v.dims and self.dims[v.dims[0]][1] == 0]
        if len(record) == 1:
            return record[0].vsize
        return sum(v.vsize for v in record)


class _Reader:
    def __init__(self, data: bytes, version: int):
        self.data, self.pos, self.version = data, 0, version

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise EOFError('En-tête NetCDF tronqué : augmenter la taille lue.')
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def int32(self) -> int:
        return struct.unpack('>i', self.take(4))[0]

    def offset(self) -> int:
        return struct.unpack('>q' if self.version == 2 else '>i', self.take(8 if self.version == 2 else 4))[0]

    def name(self) -> str:
        n = self.int32()
        raw = self.take(n)
        self.take((-n) % 4)
        return raw.decode('utf-8')

    def values(self, nc_type: int, n: int):
        code, size = _TYPES[nc_type]
        raw = self.take(n * size)
        self.take((-(n * size)) % 4)
        if nc_type == 2:
            return raw.decode('utf-8', 'replace')
        vals = struct.unpack('>' + code * n, raw)
        return vals[0] if n == 1 else vals

    def attrs(self) -> dict:
        tag, n = self.int32(), self.int32()
        if tag == 0:
            return {}
        if tag != 12:
            raise ValueError('Liste d’attributs NetCDF invalide.')
        out = {}
        for _ in range(n):
            key = self.name()
            nc_type, count = self.int32(), self.int32()
            out[key] = self.values(nc_type, count)
        return out


def parse_header(data: bytes) -> Header:
    if data[:3] != b'CDF' or data[3] not in (1, 2):
        raise ValueError('Fichier NetCDF classique (CDF-1/CDF-2) attendu.')
    r = _Reader(data, data[3])
    r.take(4)
    numrecs = r.int32()
    tag, n = r.int32(), r.int32()
    dims = [(r.name(), r.int32()) for _ in range(n)] if tag == 10 else []
    r.attrs()
    tag, n = r.int32(), r.int32()
    variables = {}
    for _ in range(n if tag == 11 else 0):
        name = r.name()
        ndims = r.int32()
        dimids = tuple(r.int32() for _ in range(ndims))
        attrs = r.attrs()
        nc_type, vsize = r.int32(), r.int32()
        variables[name] = Variable(name, dimids, attrs, nc_type, vsize, r.offset())
    return Header(data[3], numrecs, dims, variables)


def _get(url: str, start: int, end: int, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'generative-ia-research'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = resp.read()
            if len(payload) != end - start + 1:
                raise IOError(f'Réponse partielle inattendue ({len(payload)} octets).')
            return payload
        except (urllib.error.URLError, IOError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('inaccessible')


def grid_indices(lat_max: float, lat_min: float, lon_min: float, lon_max: float):
    """Indices de lignes (latitudes décroissantes) et colonnes (0–359,75°E) de la boîte."""
    rows = np.arange(round((90 - lat_max) / RES), round((90 - lat_min) / RES) + 1)
    lons = np.round(np.arange(lon_min, lon_max + RES / 2, RES), 4)
    cols = np.round(np.mod(lons, 360) / RES).astype(int)
    return rows, cols, 90 - rows * RES, lons


def read_box(url: str, hour: int, rows: np.ndarray, cols: np.ndarray, header_bytes: int = 16384) -> np.ndarray:
    """Valeurs physiques (float32) de la boîte à l’heure UTC demandée."""
    header = parse_header(_get(url, 0, header_bytes - 1))
    data_vars = [v for v in header.variables.values() if len(v.dims) == 3]
    if len(data_vars) != 1:
        raise ValueError('Une seule variable (time, latitude, longitude) attendue.')
    var = data_vars[0]
    shape = tuple(header.dims[d][1] for d in var.dims[1:])
    if shape != (NLAT, NLON) or var.nc_type != 3:
        raise ValueError(f'Grille ou type inattendu : {shape}, type {var.nc_type}.')
    row_bytes = NLON * 2
    time_len = header.dims[var.dims[0]][1]
    if time_len == 0:  # dimension illimitée : enregistrements entrelacés
        if not 0 <= hour < header.numrecs:
            raise ValueError('Heure absente du fichier.')
        step = header.record_size
    else:  # dimension fixe : tableau contigu (time, lat, lon)
        if not 0 <= hour < time_len:
            raise ValueError('Heure absente du fichier.')
        step = NLAT * row_bytes
    start = var.begin + hour * step + int(rows[0]) * row_bytes
    raw = _get(url, start, start + len(rows) * row_bytes - 1)
    packed = np.frombuffer(raw, dtype='>i2').reshape(len(rows), NLON)[:, cols]
    values = packed.astype(np.float64) * var.attrs.get('scale_factor', 1.0) + var.attrs.get('add_offset', 0.0)
    fill = {var.attrs.get('_FillValue'), var.attrs.get('missing_value')} - {None}
    if fill:
        values[np.isin(packed, list(fill))] = np.nan
    return values.astype(np.float32)


def pressure_level_url(day, variable: str, level: int) -> str:
    return f'{BASE_URL}/date-variable-pressure_level/{day:%Y/%m/%d}/{variable}/{level}.nc'


def static_url(variable: str) -> str:
    return f'{BASE_URL}/date-variable-static/2021/12/31/{variable}/static.nc'
