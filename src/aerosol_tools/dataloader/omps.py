import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


def load_omps_usask(data_dir: str | Path,
                    min_time: str | np.datetime64 | None = None,
                    max_time: str | np.datetime64 | None = None):
    """

    Parameters
    ----------
    data_dir : Path | str
        Path to monthly Usask v1.3 files
    min_time : str | np.datetime64 | None
        first time to load. Default `None` will load all available data.
    max_time : str | np.datetime64 | None
        last time to load. Default `None` will load all available data.

    Returns
    -------
    xr.Dataset
    """

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    data = []
    if type(min_time) is str:
        min_time = np.datetime64(min_time)
    if type(max_time) is str:
        max_time = np.datetime64(max_time)

    for file in data_dir.glob('*.nc'):

        # skip files outside of time bounds
        ftime = file.name.split('-')[5]
        if min_time and np.datetime64(f'{ftime[0:4]}-{ftime[4:6]}-01') + np.timedelta64(32, "D") < min_time:
            continue
        if max_time and np.datetime64(f'{ftime[0:4]}-{ftime[4:6]}-01 00:00:00') > max_time:
            continue

        try:
            tmp = xr.open_dataset(file)
        except Exception as e:
            print(e)
        else:
            data.append(tmp.sortby('time'))

    omps = xr.concat(data, dim='time').sortby('time')
    omps = omps.rename({
        'volume_extinction_coefficient_in_air_due_to_ambient_aerosol_particles': 'extinction',
        'volume_extinction_coefficient_in_air_due_to_ambient_aerosol_particles_standard_error': 'extinction_error'})
    omps = omps[['extinction', 'extinction_error', 'latitude', 'longitude', 'tropopause_altitude', 'pressure', 'temperature', 'orbit']]
    if min_time is not None:
        omps = omps.sel(time=slice(min_time, max_time))

    omps['extinction'] = omps['extinction'].where(omps.extinction_error > 0)
    omps['extinction'] = omps['extinction'].where(omps.extinction_error / omps.extinction < 1.0)

    return omps


def load_omps_iup(data_dir: str | Path,
                  min_time: str | np.datetime64 | None = None,
                  max_time: str | np.datetime64 | None = None):
    """

    Parameters
    ----------
    data_dir : Path | str
        Path to yearly v2.1 IUP files
    min_time : str | np.datetime64 | None
        first time to load. Default `None` will load all available data.
    max_time : str | np.datetime64 | None
        last time to load. Default `None` will load all available data.

    Returns
    -------
    xr.Dataset
    """

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    years = np.unique([d.year for d in pd.date_range(min_time, max_time, freq='D')])
    data = []
    for year in years:
        try:
            temp = xr.open_dataset(data_dir / f'OMPS_Limb_AER_V2_1_{year}.nc')[
                ['Aer_Extinct_Coeff', 'Solar_Zenith_Angle', 'Alt_Grid', 'Average_Latitude', 'Average_Longitude',
                 'Month', 'Day', 'UTC_Hours', 'UTC_Minutes', 'UTC_Seconds', 'AEC_Uncert']
            ].load()
        except FileNotFoundError:
            continue

        bad = temp.UTC_Hours == 24
        temp = temp.where(~bad, drop=True)
        times = [pd.Timestamp(year, int(month), int(day), int(hour), int(minute), second)
                 for month, day, hour, minute, second
                 in zip(temp.Month.to_numpy(), temp.Day.to_numpy(),
                        temp.UTC_Hours.to_numpy(), temp.UTC_Minutes.to_numpy(), temp.UTC_Seconds.to_numpy())]
        times = np.array([t.to_numpy() for t in times])
        temp['time'] = xr.DataArray(times, dims=['Num_Measurements'], coords=[temp.Num_Measurements.to_numpy()])
        temp['altitude'] = xr.DataArray(temp.Alt_Grid.values[0], dims=['Num_Alt_Levs'], coords=[temp.Num_Alt_Levs.to_numpy()])
        temp = temp.swap_dims({'Num_Alt_Levs': 'altitude', 'Num_Measurements': 'time'})
        temp = temp.rename({'Average_Latitude': 'latitude',
                            'Average_Longitude': 'longitude',
                            'Aer_Extinct_Coeff': 'extinction',
                            'AEC_Uncert': 'extinction_error',
                            'Solar_Zenith_Angle': 'SZA'})[
            ['extinction', 'extinction_error', 'SZA', 'latitude', 'longitude']
        ]
        data.append(temp.sortby('time'))

    # filter out duplicate times
    data = xr.concat(data, dim='time').sel(time=slice(min_time, max_time)).sortby('altitude')
    _, idx = np.unique(data.time.values, return_index=True)
    data = data.isel(time=idx)
    return data


def load_omps_nasa(data_dir: str | Path,
                   min_time: str | np.datetime64 | None = None,
                   max_time: str | np.datetime64 | None = None,
                   wavelength: float | None = None,
                   crosstrack: int | None = None):
    """

    Parameters
    ----------
    data_dir : str | Path
        Location of OMPS yearly files created from `aerosol_tools.dataloader.create_omps_yearly_files`
    min_time : str | np.datetime64 | None
        first time to load. Default `None` will load all available data.
    max_time : str | np.datetime64 | None
        last time to load. Default `None` will load all available data.
    wavelength : float
        select only a certain wavelength.
    crosstrack : int
        select only a particular cross track measurement

    Returns
    -------

    """
    def subselect(data):
        subset = data.sel(altitude=slice(0, 41))

        if wavelength is not None:
            subset = subset.sel(wavelength=wavelength, method='nearest')

        if crosstrack is not None:
            subset = subset.sel(crosstrack=crosstrack)

        return subset

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    ds = xr.open_mfdataset(str(data_dir / '*.nc'), preprocess=subselect)

    if min_time is not None and max_time is not None:
        ds = ds.sel(time=slice(min_time, max_time))

    ds = ds.rename({'RetrievedExtCoeff': 'extinction', 'TropopauseAltitude': 'tropopause_altitude'})
    return ds


def create_omps_yearly_files(data_dir: str | Path,
                             output_dir: str | Path,
                             start_year: int = 2011,
                             end_year: int = 2023,
                             version: str = 'v21'):
    """
    Loading NASA's daily files can be quite slow, so create yearly files better suited for xarray loading.

    Parameters
    ----------
    data_dir : Path | str
        Location of OMPS daily files
    output_dir : Path | str
        Location to save yearly files
    start_year : int
        first year to process
    end_year : int
        last year to process
    version : str
        version number to use in filenames. Default `v21`

    """

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    for file in data_dir.glob("**/*.nc"):
        folder = file.parent.name
        if (int(folder) < start_year) or (int(folder) > end_year):
            continue

        data = []

        prof = xr.open_dataset(file, group='ProfileFields')
        geo = xr.open_dataset(file, group='GeolocationFields')
        anc = xr.open_dataset(file, group='AncillaryData')
        ds = xr.merge([prof, geo, anc])
        time = pd.Timestamp(str(int(ds.Date.values))) + ds.SecondsInDay
        ds['DimAlongTrack'] = time
        ds['DimAltitudeLevel'] = ds.Altitude
        ds['DimWavelengthRetGrid'] = ds.Wavelength
        ds = ds[['RetrievedExtCoeff', 'RetrievedExtCoeff_NOFILT',
                 'ExtCoeffError', 'CloudHeight', 'Latitude',
                 'Longitude', 'SingleScatteringAngle', 'SolarZenithAngle',
                 'CloudType', 'TropopauseAltitude', 'Residual']]
        data.append(ds)

        data = xr.concat(data, dim='DimAlongTrack')
        data = data.rename({'DimAlongTrack': 'time', 'DimAltitudeLevel': 'altitude',
                            'DimWavelengthRetGrid': 'wavelength', 'DimCrossTrack': 'crosstrack',
                            'Latitude': 'latitude', 'Longitude': 'longitude'}).sortby('time')
        data.to_netcdf(output_dir / f'omps_nasa_{folder}_aer_{version}.nc')
