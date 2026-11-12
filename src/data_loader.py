"""Indice C-Haines (Mills & McCaw), avec point de rosée approché par Magnus."""
import numpy as np
import xarray as xr

C_HAINES_REFERENCE='https://library.dbca.wa.gov.au/FullTextFiles/921091.pdf'
MULTIYEAR_PATH='data/era5_firms_jjas_2001_2024.nc'
# Niveau 850 hPa ≈ 1 500 m en été : au-delà de ce relief moyen de maille, il peut être sous la surface.
MAX_ELEVATION_850=1200.

def _open(path):
    """Ouverture par objet fichier : robuste aux chemins Windows accentués (bibliothèque C netCDF)."""
    with open(path,'rb') as handle, xr.open_dataset(handle,engine='h5netcdf') as ds:return ds.load()

def _haines_from_dewpoint(t850_c,t700_c,dewpoint):
    depression=np.minimum(t850_c-dewpoint,30.)
    ca=.5*(t850_c-t700_c)-2.
    cb=depression/3.-1.
    cb=xr.where(cb>5,5+(cb-5)/2,cb) if isinstance(cb,xr.DataArray) else np.where(cb>5,5+(cb-5)/2,cb)
    return ca+cb

def continuous_haines(t850_c,t700_c,rh850_percent):
    rh=rh850_percent
    invalid=(rh<=0)|(rh>100)|~np.isfinite(rh)
    if bool(invalid.any()):raise ValueError('Humidité relative attendue dans ]0,100] %, sans valeur non finie.')
    gamma=np.log(rh/100.)+17.625*t850_c/(243.04+t850_c)
    dewpoint=243.04*gamma/(17.625-gamma)
    return _haines_from_dewpoint(t850_c,t700_c,dewpoint)

def dewpoint_from_q(q,pressure_hpa):
    """Point de rosée (°C) depuis l'humidité spécifique (kg/kg) : e = q p / (0,622 + 0,378 q), Magnus inversé."""
    q=np.asarray(q,dtype=float)
    if (q<=0).any() or not np.isfinite(q).all():raise ValueError('Humidité spécifique strictement positive et finie requise.')
    e=q*pressure_hpa/(.622+.378*q)
    g=np.log(e/6.112)
    return 243.04*g/(17.625-g)

def continuous_haines_from_q(t850_c,t700_c,q850,pressure_hpa=850.):
    """C-Haines sans humidité relative : pas de troncature RH ni de sursaturation à masquer."""
    return _haines_from_dewpoint(np.asarray(t850_c,dtype=float),np.asarray(t700_c,dtype=float),dewpoint_from_q(q850,pressure_hpa))

def load_haines(path, *, method='continuous'):
    ds=_open(path)
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

def load_multiyear(path=MULTIYEAR_PATH,*,max_elevation=MAX_ELEVATION_850,min_land=.5):
    """Jeu ERA5 + FIRMS : mailles terrestres sous le relief limite seulement."""
    ds=_open(path)
    if not {'haines_index','fire_count','land_sea_mask','surface_elevation'}.issubset(ds.data_vars):raise ValueError('Jeu multi-annuel incomplet.')
    keep=(ds.land_sea_mask>=min_land)&(ds.surface_elevation<=max_elevation)
    index=ds.haines_index.where(keep);index.name='haines_index'
    index.attrs={'definition':'C-Haines, dewpoint from specific humidity','source':C_HAINES_REFERENCE,
                 'cells_kept':int(keep.sum()),'cells_total':int(keep.size),'max_elevation_m':max_elevation,'min_land_fraction':min_land}
    return ds,index,keep
