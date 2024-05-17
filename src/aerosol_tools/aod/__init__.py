from __future__ import annotations

import numpy as np
import xarray as xr

from aerosol_tools import truncate_below_tropopause


def calculate_aod_cdf_tropopause(data: xr.Dataset, max_alt_km: float = 35.0):
    """

    Parameters
    ----------

    data : xr.Dataset
        Must have `extinction`, `tropopause_altitude` and `altitude` fields.
    max_alt_km : float
        Maximum altitude in km to use in calculation.

    Returns
    -------
    xr.Dataset
    """

    data = truncate_below_tropopause(data, fill_value=0.0)
    mean_extinction = data.extinction.mean(dim="time")
    aod = mean_extinction.sel(altitude=slice(0, max_alt_km)).sum(dim="altitude")
    return aod.rename("AOD").to_dataset()


def calculate_aod_average_tropopause(data: xr.Dataset, max_alt_km: float = 35.0):
    """

    Parameters
    ----------

    data : xr.Dataset
        Must have `extinction` and `tropopause_altitude` fields.
    max_alt_km : float
        Maximum altitude in km to use in calculation.

    Returns
    -------
    xr.Dataset
    """

    mean_trop = data.tropopause_altitude.mean(dim="time")
    data["extinction"] = data.extinction.where(data.altitude > mean_trop)
    mean_extinction = data.extinction.mean(dim="time")
    aod = mean_extinction.sel(altitude=slice(0, max_alt_km)).sum(dim="altitude")
    return aod.rename("AOD").to_dataset()


def calculate_aod_local_tropopause(data: xr.Dataset, max_alt_km: float = 35.0):
    """

    Parameters
    ----------

    data : xr.Dataset
        Must have `extinction` and `tropopause_altitude` fields.
    max_alt_km : float
        Maximum altitude in km to use in calculation.

    Returns
    -------
    xr.Dataset
    """

    data = truncate_below_tropopause(data, fill_value=np.nan)
    mean_extinction = data.extinction.mean(dim="time")
    aod = mean_extinction.sel(altitude=slice(0, max_alt_km)).sum(dim="altitude")
    return aod.rename("AOD").to_dataset()


def calculate_aod_per_profile(data: xr.Dataset, max_alt_km: float = 35.0):
    """

    Parameters
    ----------

    data : xr.Dataset
        Must have `extinction` and `tropopause_altitude` fields.
    max_alt_km : float
        Maximum altitude in km to use in calculation.

    Returns
    -------
    xr.Dataset
    """

    data = truncate_below_tropopause(data, fill_value=np.nan)
    aod = (
        data.extinction.sel(altitude=slice(0, max_alt_km))
        .sum(dim="altitude")
        .mean(dim="time")
    )
    return aod.rename("AOD").to_dataset()
