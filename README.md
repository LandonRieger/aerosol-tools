# Aerosol Tools

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/LandonRieger/aerosol_tools/main.svg)](https://results.pre-commit.ci/latest/github/LandonRieger/aerosol_tools/main)

Python tools to load and process OMPS aerosol data

## Installation
The package can be installed through `pip` with

```bash
pip install aerosol_tools@git+https://github.com/LandonRieger/aerosol-tools
```

## Usage

### Loading data

```python
from aerosol_tools import load_omps_usask

omps = load_omps_usask('path\\to\\data')
omps.extinction.where((omps.latitude < 10) &
                      (omps.latitude > -10))\
               .resample(time='1MS')\
               .mean()\
               .plot(x='time')
```

### Computing Aerosol Optical Depth

```python
from aerosol_tools import load_omps_usask, calculate_aod_cdf_tropopause

omps = load_omps_usask('path\\to\\data')
ds = omps.extinction.where((omps.latitude < 10) &
                           (omps.latitude > -10))\
                    .sel(time=slice('2020-01-01', '2020-01-31'))
aod = calculate_aod_cdf_tropopause(ds)
```


## License
This project is licensed under the MIT license
