import numpy as np
import pandas as pd
import pytest
from src.data_loader import continuous_haines,load_haines
from src.preprocessing import temporal_split,fit_transform,transform

def test_haines_saturated_and_dry_formula():
 assert continuous_haines(np.array([20.]),np.array([10.]),np.array([100.]))[0]==pytest.approx(2.)
 assert continuous_haines(np.array([20.]),np.array([10.]),np.array([1.]))[0]==pytest.approx(10.)
 with pytest.raises(ValueError):continuous_haines(np.array([20.]),np.array([10.]),np.array([0.]))

def test_actual_source_invalid_rh_is_counted_and_masked():
 ds,h=load_haines('data/haines_data.nc')
 rh=ds.r.sel(pressure_level=850);invalid=((rh<=0)|(rh>100)).values
 assert int(h.isnull().sum())==int(invalid.sum())>0
 assert h.attrs['excluded_invalid_rh_or_temperature']==int(invalid.sum())
 assert np.isnan(h.values[invalid]).all()

def frame():
 return pd.DataFrame({'valid_time':np.repeat(pd.date_range('2023-07-01',periods=20),2),'latitude':[45,46]*20,'longitude':[1,2]*20,'haines_index':np.arange(40,dtype=float)})

def test_dates_are_grouped_and_scaling_is_train_only():
 train,val,test=temporal_split(frame())
 assert set(train.valid_time).isdisjoint(test.valid_time)
 assert (val.valid_time.min()-train.valid_time.max()).days>=2
 assert (test.valid_time.min()-val.valid_time.max()).days>=2
 train,lam,scaler,shift=fit_transform(train)
 assert lam is None and shift==0
 mean=scaler.mean_.copy();heldout=test.copy();heldout['haines_index']=10000
 transformed=transform(heldout,scaler)
 np.testing.assert_equal(mean,scaler.mean_)
 assert transformed.haines_final.min()>1
 np.testing.assert_allclose(scaler.inverse_transform(transformed[['haines_final']].to_numpy()).ravel(),heldout.haines_index)

def test_short_calendar_or_constant_target_rejected():
 with pytest.raises(ValueError):temporal_split(frame().iloc[:4])
 f=frame();f.haines_index=1
 with pytest.raises(ValueError):fit_transform(f)

def test_negative_embargo_is_rejected():
 with pytest.raises(ValueError,match='Embargo'):temporal_split(frame(),gap_days=-1)
