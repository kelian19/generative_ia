"""Indice C-Haines (Mills & McCaw), avec point de rosée approché par Magnus."""
import numpy as np
import xarray as xr

C_HAINES_REFERENCE='https://library.dbca.wa.gov.au/FullTextFiles/921091.pdf'

def continuous_haines(t850_c,t700_c,rh850_percent):
    rh=rh850_percent
    invalid=(rh<=0)|(rh>100)|~np.isfinite(rh)
    if bool(invalid.any()):raise ValueError('Humidité relative attendue dans ]0,100] %, sans valeur non finie.')
    gamma=np.log(rh/100.)+17.625*t850_c/(243.04+t850_c)
    dewpoint=243.04*gamma/(17.625-gamma)
    depression=np.minimum(t850_c-dewpoint,30.)
    ca=.5*(t850_c-t700_c)-2.
    cb=depression/3.-1.
    cb=xr.where(cb>5,5+(cb-5)/2,cb) if isinstance(cb,xr.DataArray) else np.where(cb>5,5+(cb-5)/2,cb)
    return ca+cb

def load_haines(path, *, method='continuous'):
    with xr.open_dataset(path) as handle:ds=handle.load()
    if not {'t','r'}.issubset(ds.data_vars):raise ValueError('Champs t et r requis.')
    if 'pressure_level' not in ds.coords or not {700.,850.}.issubset(ds.pressure_level.values):raise ValueError('Niveaux 700 et 850 hPa requis.')
    if ds.t.attrs.get('units')!='K' or ds.r.attrs.get('units')!='%':raise ValueError('Unités attendues : température en K, humidité en %.')
    t850=ds.t.sel(pressure_level=850)-273.15;t700=ds.t.sel(pressure_level=700)-273.15;rh=ds.r.sel(pressure_level=850)
    if method=='continuous':
        valid=np.isfinite(t850)&np.isfinite(t700)&np.isfinite(rh)&(rh>0)&(rh<=100)
        # Invalid RH is masked, not clipped into a false valid observation.
        safe_rh=xr.where(valid,rh,100.)
        index=continuous_haines(t850,t700,safe_rh).where(valid)
        invalid_count=int((~valid).sum())
    elif method=='legacy_proxy':index=t850-t700+(100-rh)/5
    else:raise ValueError('Méthode inconnue.')
    index.name='haines_index';index.attrs={'definition':'C-Haines / Magnus approximation' if method=='continuous' else 'Legacy proxy: NOT C-Haines','source':C_HAINES_REFERENCE,'excluded_invalid_rh_or_temperature':invalid_count if method=='continuous' else 0}
    return ds,index
