import pytest
from aerosol_tools import load_omps_usask, load_omps_nasa, load_omps_iup
from pathlib import Path


def test_usask():
    data = load_omps_usask(Path(r'C:\Users\lar555\data\omps\usask\aerosol\v1.3\aerosol\monthly'),
                           min_time='2020-01-01',
                           max_time='2020-02-01')
    assert len(data.time) == 67964


def test_nasa():
    data = load_omps_nasa(Path(r'C:\Users\lar555\data\omps\nasa\aerosol\yearly\v21'),
                          min_time='2020-01-01',
                          max_time='2020-02-01')
    assert len(data.time) == 71487


def test_nasa_crosstrack():
    data = load_omps_nasa(Path(r'C:\Users\lar555\data\omps\nasa\aerosol\yearly\v21'),
                          min_time='2020-01-01',
                          max_time='2020-02-01',
                          crosstrack=1)
    assert data.latitude.shape == (71487,)


def test_iup():
    data = load_omps_nasa(Path(r'C:\Users\lar555\data\omps\iup'),
                          min_time='2020-01-01',
                          max_time='2020-02-01')
    assert len(data.time) == 71487
