from __future__ import annotations

from pathlib import Path

import pytest

from aerosol_tools import (
    calculate_aod_cdf_tropopause,
    calculate_aod_local_tropopause,
    calculate_aod_mean_tropopause,
    calculate_aod_per_profile,
    load_omps_usask,
)

AOD_PRECISION = 0.0000001


@pytest.fixture()
def omps_data():
    ds = load_omps_usask(
        Path(r"C:\Users\lar555\data\omps\usask\aerosol\v1.3\aerosol\monthly"),
        min_time="2020-01-01",
        max_time="2020-02-01",
    )
    return ds.where((ds.latitude > -10) & (ds.latitude < 10), drop=True)


def test_aod_cdf(omps_data):
    aod = calculate_aod_cdf_tropopause(omps_data)
    assert pytest.approx(float(aod.AOD.to_numpy()), AOD_PRECISION) == 0.0061631862


def test_aod_mean_trop(omps_data):
    aod = calculate_aod_mean_tropopause(omps_data)
    assert pytest.approx(float(aod.AOD.to_numpy()), AOD_PRECISION) == 0.0056597978


def test_aod_local_trop(omps_data):
    aod = calculate_aod_local_tropopause(omps_data)
    assert pytest.approx(float(aod.AOD.to_numpy()), AOD_PRECISION) == 0.0079408503


def test_aod_per_profile(omps_data):
    aod = calculate_aod_per_profile(omps_data)
    assert pytest.approx(float(aod.AOD.to_numpy()), AOD_PRECISION) == 0.0061631862
