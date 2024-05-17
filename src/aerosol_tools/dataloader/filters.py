from __future__ import annotations

import numpy as np
import xarray as xr


def truncate_below_max_value(data: xr.Dataset, filter_val: float):
    """
    Filter all points below an extinction value that exceeds `filter_val`

    Parameters
    ----------
    data : xr.Dataset
        Must have `extinction` and `altitude` fields.
    filter_val : float
        Maximum value of extinction.

    Returns
    -------
    xr.Dataset
    """

    min_alt = (data.extinction >= filter_val) * data.altitude
    min_alt = min_alt.where(min_alt > 0).max(dim="altitude") + 0.1
    min_alt = min_alt.fillna(0.0)
    data["extinction"] = data.extinction.where(data.altitude > min_alt)

    return data


def truncate_below_tropopause(
    data: xr.Dataset, km_above: float = 0.0, fill_value: float | None = None
):
    """
    Filter all points below the tropopause and replace with `fill_value`

    Parameters
    ----------
    data : xr.Dataset
        Must have `extinction`, `tropopause_altitude` and `altitude` fields.
    km_above : float
        Shift tropopause up by `km_above`.
    fill_value : float
        Fill values below the tropause with `fill_value`.

    Returns
    -------
    xr.Dataset
    """

    if fill_value is None:
        fill_value = np.nan

    data["extinction"] = data.extinction.where(
        data.altitude > data.tropopause_altitude + km_above, other=fill_value
    )
    return data
