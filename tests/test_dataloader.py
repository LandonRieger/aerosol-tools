from __future__ import annotations

from pathlib import Path

import pytest

from aerosol_tools import load_omps_iup, load_omps_nasa, load_omps_usask


@pytest.mark.skip("local test only")
def test_usask():
    data = load_omps_usask(
        Path(r"C:\Users\lar555\data\omps\usask\aerosol\v1.3\aerosol\monthly"),
        start_date="2020-01-01",
        end_date="2020-02-01",
    )
    assert len(data.time) == 67964


@pytest.mark.skip("local test only")
def test_nasa():
    data = load_omps_nasa(
        Path(r"C:\Users\lar555\data\omps\nasa\aerosol\yearly\v21"),
        start_date="2020-01-01",
        end_date="2020-02-01",
    )
    assert len(data.time) == 71487


@pytest.mark.skip("local test only")
def test_nasa_crosstrack():
    data = load_omps_nasa(
        Path(r"C:\Users\lar555\data\omps\nasa\aerosol\yearly\v21"),
        start_date="2020-01-01",
        end_date="2020-02-01",
        crosstrack=1,
    )
    assert data.latitude.shape == (71487,)


@pytest.mark.skip("local test only")
def test_iup():
    data = load_omps_iup(
        Path(r"C:\Users\lar555\data\omps\iup"),
        start_date="2020-01-01",
        end_date="2020-02-01",
    )
    assert len(data.time) == 58176
