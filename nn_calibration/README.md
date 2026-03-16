# Calibration Code

## Installation

A python package implements all parts dicussed int he previous section. To run
it, first install the package.
```bash
git clone REPO
cd REPO
python -m pip install .
```

## Running

To find clusters
```bash
booms_tools nn_calibration cluster PATH_TO_CLUSTER
```

To form a basis
```bash
booms_tools nn_calibration basis -l PATH_TO_LIST_TXT
```

To calibrate
To form a basis
```bash
booms_tools nn_calibration calibration -l PATH_TO_LIST_TXT
```

## Sidecar files
