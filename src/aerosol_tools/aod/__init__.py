from __future__ import annotations

import numpy as np
import xarray as xr

from aerosol_tools.filters import truncate_below_tropopause


def calculate_aod_cdf_tropopause(
    data: xr.Dataset, max_alt_km: float = 35.0, km_above: float = 0.0
):
    r"""
    Aerosol optical depth (AOD) computed using the mean extinction profile weighted by the
    probability that the altitude is in the stratosphere. This helps avoid the issue of
    the occasional low-altitude tropopause biasing the AOD to low altitudes that are not
    representative of the bin.

    NOTE: This works well for well-sampled conditions, but should use an external tropopause
    if samples are not representative of the bin (i.e. sparse measurements such as SAGE).

    Given N tropopause samples, the cumalative distribution function of the tropopause
    altitude is given as:

    .. math::
        CDF(z) = \frac{1}{N}\sum_{i=1}^{N}1_{Z_i \leq z}


    The mean extinction profile is calculated by setting all values below the local
    tropopause to NaN and taking the average at each altitude, ignoring NaNs. The
    AOD is then given as:

    .. math::
        AOD = \int_{tropopause}^{max alt km} k_{mean}(z')CDF(z')dz'

    Parameters
    ----------

    data : xr.Dataset
        Must have `extinction`, `tropopause_altitude` and `altitude` fields.
    max_alt_km : float
        Maximum altitude in km to use in calculation.
    km_above : float
        Shift tropopause up by `km_above`.

    Returns
    -------
    xr.Dataset
    """

    data = truncate_below_tropopause(data, km_above=km_above, fill_value=0.0)
    mean_extinction = data.extinction.mean(dim="time").sel(
        altitude=slice(0, max_alt_km)
    )
    aod = mean_extinction.integrate("altitude")
    return aod.rename("AOD").to_dataset()


def calculate_aod_mean_tropopause(
    data: xr.Dataset, max_alt_km: float = 35.0, km_above: float = 0.0
):
    """
    Use the mean extinction profile truncated at the mean tropopause to
    compute the AOD.

    Parameters
    ----------

    data : xr.Dataset
        Must have `extinction` and `tropopause_altitude` fields.
    max_alt_km : float
        Maximum altitude in km to use in calculation.
    km_above : float
        Shift tropopause up by `km_above`.

    Returns
    -------
    xr.Dataset
    """

    mean_trop = data.tropopause_altitude.mean(dim="time")
    data["extinction"] = data.extinction.where(data.altitude > mean_trop + km_above)
    mean_extinction = data.extinction.mean(dim="time").sel(
        altitude=slice(0, max_alt_km)
    )
    aod = mean_extinction.where(mean_extinction > 0, drop=True).integrate("altitude")
    return aod.rename("AOD").to_dataset()


def calculate_aod_local_tropopause(
    data: xr.Dataset, max_alt_km: float = 35.0, km_above: float = 0.0
):
    """
    Use the mean extinction profile truncated at the local tropopause to
    compute the AOD.

    Parameters
    ----------

    data : xr.Dataset
        Must have `extinction` and `tropopause_altitude` fields.
    max_alt_km : float
        Maximum altitude in km to use in calculation.
    km_above : float
        Shift tropopause up by `km_above`.

    Returns
    -------
    xr.Dataset
    """

    data = truncate_below_tropopause(data, km_above=km_above, fill_value=np.nan)
    mean_extinction = data.extinction.mean(dim="time").sel(
        altitude=slice(0, max_alt_km)
    )
    aod = mean_extinction.where(mean_extinction > 0, drop=True).integrate("altitude")
    return aod.rename("AOD").to_dataset()


def calculate_aod_per_profile(
    data: xr.Dataset, max_alt_km: float = 35.0, km_above: float = 0.0
):
    """
    Use each extinction profile to compute the AOD and then average. This is equivalent to
    `calculate_aod_cdf_tropopause` if all profiles extend to the tropopause but will tend to
    underestimate in the case of missing data.


    Parameters
    ----------

    data : xr.Dataset
        Must have `extinction` and `tropopause_altitude` fields.
    max_alt_km : float
        Maximum altitude in km to use in calculation.
    km_above : float
        Shift tropopause up by `km_above`.

    Returns
    -------
    xr.Dataset
    """

    data = truncate_below_tropopause(data, km_above=km_above, fill_value=np.nan)
    aod = (
        data.extinction.sel(altitude=slice(0, max_alt_km))
        .fillna(0.0)
        .integrate("altitude")
        .mean(dim="time")
    )
    return aod.rename("AOD").to_dataset()
