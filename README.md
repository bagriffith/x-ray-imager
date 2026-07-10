# X-ray Imager Processing

<!-- start overview -->
This package contains three sets of tools needed to calibrate and interpret
scintillator-based x-ray imagers.

It is primarily for the [BOOMS](https://ssel.montana.edu/projects/booms.html)
imagers. They use a monolithic NaI(Tl) crystal read out by four square
photomultiplier tubes. In places, this package assumes this geometry, but
comments indicate where.

[Documentation](https://x-ray-imager.github.bradyagriffith.com/)
<!-- end overview -->

## Component Overview

### Identify Gamma-ray Calibration Lines

Each imager was calibrated using a set of gamma-ray sources by recording
the response with the source at each point in a square grid. A clustering
algorithm separates source x-rays from background.

There are two CLI tools under `identify-lines` for this calibration.

- `single` processes a list of detected x-rays and returns a mean response to
  each major energy line. The source position should be constant.
- `multiple` processes several lists, returning the response for each. Use this
  to process a grid of calibration points.

### Response Interpolation

This takes a grid of positions/energies responses and returns a best estimate
for the response at intermediate positions/energies. The CLI for this is
`response-interpolation`.

### Position Estimation

A series of x-ray imager observations will have an estimated position and
energy assigned based on an interpolated calibration response. The CLI for
this is `position-estimation`.

## Installation

<!-- start installation -->
Installation has only been tested on Ubuntu 24.04 with Python 3.12 but should
be compatible with most systems. To install the current version, run the
command below.

```bash
pip install https://github.com/bagriffith/x-ray-imager/archive/refs/heads/main.zip
```
<!-- end installation -->
