from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aerosol_tools import (
    calculate_aod_cdf_tropopause,
    calculate_aod_local_tropopause,
    calculate_aod_mean_tropopause,
    calculate_aod_per_profile,
)

AOD_PRECISION = 0.0000001


"""
Test AOD calculation

 - 10 profiles with tropopause linearly increasing from 10.5 to 15km.
 - Extinction is 1.0 at all points.
 - Highest altitude of 34.5km.
 - Two profiles have retrievals that truncate above the tropopause simulating saturation.
 - Theoretically correct AOD is 21.75
"""


@pytest.fixture()
def omps_data():

    time = pd.date_range("2020-01-01 4:00:00", periods=10, freq="10s")
    alt = np.arange(10.5, 40.5)
    ext = np.ones((len(time), len(alt)))
    ext[7, 0:10] = np.nan
    ext[3, 0:2] = np.nan
    tropopause = np.linspace(10.5, 15.0, len(time))
    return xr.Dataset(
        {
            "extinction": (["time", "altitude"], ext),
            "tropopause_altitude": (["time"], tropopause),
        },
        coords={"time": time, "altitude": alt},
    )


def test_aod_cdf(omps_data):
    aod = calculate_aod_cdf_tropopause(omps_data)
    assert pytest.approx(float(aod.AOD.to_numpy()), AOD_PRECISION) == 21.477777777777778


def test_aod_mean_trop(omps_data):
    aod = calculate_aod_mean_tropopause(omps_data)
    assert pytest.approx(float(aod.AOD.to_numpy()), AOD_PRECISION) == 21.0


def test_aod_local_trop(omps_data):
    aod = calculate_aod_local_tropopause(omps_data)
    assert pytest.approx(float(aod.AOD.to_numpy()), AOD_PRECISION) == 23.0


def test_aod_per_profile(omps_data):
    aod = calculate_aod_per_profile(omps_data)
    assert pytest.approx(float(aod.AOD.to_numpy()), AOD_PRECISION) == 20.9
