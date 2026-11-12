import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.io import netcdf_file

from src.data_loader import continuous_haines, continuous_haines_from_q, dewpoint_from_q
from src.multiyear import build_frame, fire_lift, year_split, yearly_association
from src.sources import era5_raw, firms


def test_dewpoint_from_specific_humidity_round_trips_magnus():
    t = np.array([5., 20., 30.])
    e = 6.112 * np.exp(17.625 * t / (243.04 + t))  # pression de vapeur saturante à T : point de rosée = T
    q = .622 * e / (850. - .378 * e)
    np.testing.assert_allclose(dewpoint_from_q(q, 850.), t, atol=1e-9)
    with pytest.raises(ValueError):
        dewpoint_from_q(np.array([0.]), 850.)


def test_haines_from_q_matches_relative_humidity_formulation_at_saturation():
    t850, t700 = np.array([20.]), np.array([10.])
    e = 6.112 * np.exp(17.625 * 20. / (243.04 + 20.))
    q = .622 * e / (850. - .378 * e)
    assert continuous_haines_from_q(t850, t700, q)[0] == pytest.approx(continuous_haines(t850, t700, np.array([100.]))[0], abs=1e-9)


def test_era5_partial_reader_recovers_packed_values(tmp_path, monkeypatch):
    path = tmp_path / 't.nc'
    raw = np.arange(2 * era5_raw.NLAT * era5_raw.NLON, dtype=np.int32).reshape(2, era5_raw.NLAT, era5_raw.NLON) % 3000 - 1500
    with netcdf_file(path, 'w', version=2) as nc:
        nc.createDimension('longitude', era5_raw.NLON)
        nc.createDimension('latitude', era5_raw.NLAT)
        nc.createDimension('time', 2)
        v = nc.createVariable('t', 'h', ('time', 'latitude', 'longitude'))
        v[:] = raw.astype('>i2')
        v.scale_factor = .01
        v.add_offset = 270.
    data = path.read_bytes()
    calls = []

    def fake_get(url, start, end, retries=5):
        calls.append(end - start + 1)
        return data[start:end + 1]

    monkeypatch.setattr(era5_raw, '_get', fake_get)
    rows, cols, lat, lon = era5_raw.grid_indices(51., 50., -5., -4.)
    out = era5_raw.read_box('unused', 1, rows, cols)
    assert out.shape == (len(rows), len(cols)) == (5, 5)
    expected = raw[1][np.ix_(rows, cols)] * .01 + 270.
    np.testing.assert_allclose(out, expected, atol=1e-4)
    assert sum(calls) < len(data) / 50  # lecture partielle : en-tête + quelques lignes, pas le fichier entier
    np.testing.assert_allclose(lat, [51., 50.75, 50.5, 50.25, 50.])
    with pytest.raises(ValueError):
        era5_raw.read_box('unused', 2, rows, cols)


def test_grid_indices_wrap_negative_longitudes():
    rows, cols, lat, lon = era5_raw.grid_indices(51., 51., -.25, .25)
    assert cols.tolist() == [1439, 0, 1]


def _frame():
    days = pd.date_range('2001-06-01', '2024-08-31', freq='D')
    days = days[days.month.isin([6, 7, 8])][::7]
    index = xr.DataArray(np.random.default_rng(0).normal(4, 2, (len(days), 3, 4)), dims=('valid_time', 'latitude', 'longitude'),
                         coords={'valid_time': days, 'latitude': [45., 44.75, 44.5], 'longitude': [0., .25, .5, .75]})
    fires = (np.random.default_rng(1).uniform(size=index.shape) > 1.2).astype(np.uint8)
    ds = xr.Dataset({'fire_count': (index.dims, fires)}, coords=index.coords)
    keep = xr.DataArray(np.ones((3, 4), bool), dims=('latitude', 'longitude'), coords={'latitude': index.latitude, 'longitude': index.longitude})
    keep[0, 0] = False
    return build_frame(index.where(keep), ds, keep), len(days)


def test_frame_excludes_masked_cells_and_split_is_by_whole_years():
    frame, n_days = _frame()
    assert len(frame) == n_days * 11
    train, val, test = year_split(frame)
    assert train.year.max() < val.year.min() and val.year.max() < test.year.min()
    assert len(train) + len(val) + len(test) == len(frame)
    assert set(train.valid_time).isdisjoint(val.valid_time) and set(val.valid_time).isdisjoint(test.valid_time)
    with pytest.raises(ValueError):
        year_split(frame, train=(2001, 2016), validation=(2015, 2020), test=(2021, 2024))


def test_fire_lift_counts_and_yearly_association():
    frame = pd.DataFrame({'haines_index': [1., 2., 8., 9., 9., 3.], 'fire_count': [0, 0, 1, 2, 0, 0], 'year': [2001, 2001, 2002, 2002, 2003, 2003]})
    lift = fire_lift(frame, 6., bins=[-np.inf, 5, np.inf])
    assert lift['exceedance_rate'] == pytest.approx(3 / 6)
    assert lift['fire_rate_above'] == pytest.approx(2 / 3) and lift['fire_rate_below'] == pytest.approx(0.)
    assert lift['hit_rate'] == pytest.approx(1.) and lift['false_alarm_rate'] == pytest.approx(1 / 3)
    assert lift['lift'] == pytest.approx((2 / 3) / (2 / 6))
    assert [b['fires'] for b in lift['by_bin']] == [0, 2]
    table, rho, _ = yearly_association(frame, 6.)
    assert table.exceedance_cell_days.tolist() == [0, 2, 1] and table.fire_detections.tolist() == [0, 3, 0]
    assert np.isnan(rho)  # moins de quatre années : pas de corrélation de rang


def test_firms_filter_keeps_vegetation_fires_with_confidence_only():
    df = pd.DataFrame({'type': [0, 0, 2, 3], 'confidence': [80, 10, 90, 90], 'latitude': 1., 'longitude': 1., 'acq_date': '2020-07-01'})
    kept = firms.filter_vegetation(df)
    assert kept.index.tolist() == [0]


def test_firms_grid_counts_assigns_nearest_cell_and_ignores_outside():
    dates = pd.DatetimeIndex(['2020-07-01', '2020-07-02'])
    lat, lon = np.array([45., 44.75, 44.5]), np.array([0., .25, .5])
    df = pd.DataFrame({'latitude': [44.76, 44.76, 50.], 'longitude': [.26, .26, 0.], 'acq_date': ['2020-07-02', '2020-07-02', '2020-07-01']})
    counts = firms.grid_counts(df, dates, lat, lon)
    assert counts[1, 1, 1] == 2 and counts.sum() == 2
