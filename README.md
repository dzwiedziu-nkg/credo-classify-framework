# The credo-classify-framework project

Framework to analysis the [images](https://arxiv.org/pdf/1909.01929.pdf)
with cropped [cosmic-ray](https://en.wikipedia.org/wiki/Cosmic_ray) based noise
on [CMOS/CCD camera](https://en.wikipedia.org/wiki/Active-pixel_sensor).

Framework is designed for images stored in [CREDO Project database](https://api.credo.science/).  

## Requirements

The CREDO account with privileges to download data is recommended.
Without it you will work on our examples only. 

You can create new account by [CREDO Detector App](https://play.google.com/store/apps/details?id=science.credo.mobiledetector).
Next, please contact with [CREDO maintainers](https://credo.science/) to activate privileges to download.

Another requirements is a installed python v3.6 or newer.

## Getting started

Prepare software to working with cosmic-ray images:

1. Prepare python's [virtual environment](https://docs.python.org/3/library/venv.html) and activate it.
2. Install [JupyterLab](https://jupyter.org/install.html).
3. Install `credo_cf` package from GitHub.
4. Download data from CREDO database by
 [CREDO API Tools](https://github.com/credo-science/credo-api-tools)
 using your login and password to CREDO account.

### tl;dr

```bash
# Prepare venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install JupyterLab
pip install wheel
pip install jupyterlab

```


## Getting started


