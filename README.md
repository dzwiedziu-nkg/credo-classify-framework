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

Tested on Linux Mint 19.3:

```bash
# Prepare venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install JupyterLab
pip install wheel
pip install jupyterlab

# Install credo_cf package
pip install git+https://github.com/dzwiedziu-nkg/credo-classify-framework.git
## for upgrade just installed package:
# pip install --upgrade git+https://github.com/dzwiedziu-nkg/credo-classify-framework.git

# Launch JupyterLab
jupyter lab
```

Download sample data:

```bash
# Download small example (42 images)
wget -O- http://mars.iti.pk.edu.pl/~nkg/export_1585398647736_1585402962221.json.bz2 | bzip2 -dc > small_example.json 

# Download big example (80K images)
wget -O- wget http://mars.iti.pk.edu.pl/~nkg/export_1584805204914_1585394157807.json.bz2 | bzip2 -dc > big_example.json
```

The `export_*.json` files is a output files from [CREDO API Tools](https://github.com/credo-science/credo-api-tools).

## Working with Jupyter Lab

See: [example/jupyter.ipynb](example/jupyter.ipynb) for example hot to use it with Jupyter Lab.

### General conventions

The `credo_cf` is the set of functions (not classes).
Functions are working with cosmic-ray detections.
Each detection was stored in separated dict object (key-value pair map) named detection.
Functions working with single detection or list of detections. The input of functions is
the values stored in keys. The output of functions is the values stored in new keys in detection.
The docstring documentation of each function describes which keys used and stored. 

Framework contains classify functions, load/write (from JSON) functions and helpers like grouping functions.

Each classify function get single detection or list of detections as 1st arg and other
parameters in next args.
* when function gets list of detections then return tuple of two lists:
  * classified and
  * no classified
* when functions get single detection then return:
  * `True` - when classified and
  * `False` - when no classified.
  
Store the return of classify functions is not mandatory because result was stored in key value anyway.
But, the return of classify functions may be used for classify chains. 

Some classify functions should be get list of detections for the same device.
So the grouping functions is provided. Each grouping functions return dict object where
the key is the group (i.e. `device_id`) and value is the list of detections.

**Important!** Each list of detections contains reference to detection dict object.
When you modify some key value then it will be modified everywhere.
When you want to get true copy of object you should use deep copy.    

## Contribute

Welcome to contribute to write self extension to our framework.

1. Please fork project to self GitHub account.
2. Clone git repository to your local machine.
3. Create new python project with own **venv**.
4. Install `credo_cf` package from local directory via `pip install -e /path/to/cloned/credo_cf`
5. Commit and push to GitHub changes in `credo_cf`.
6. Make pull request of these changes to us.

### Project structure

The `credo_cf` is the set of functions (not classes) grouped by sub-packages.
Each function have unique name and is exported in main `credo_cf` package.

* `credo_cf/commons` - utils functions used in various other functions,
  * `consts.py` - keys for values stored in objects,
  * `grouping.py` - functions for grouping objects,
  * `classify.py` - utils functions for classification functions,
  * `utils.py` - other utils,
* `credo_cf/image` - utils functions used for working with cosmic-ray image,
* `credo_cf/io` - load and save functions,
* `credo_cf/classificaton` - function to classify cosmic-ray detections,
  * `artifact` - functions to classify as artifact
  
### Contribute rules for pull requests

Changes in pull request should be well written. 

* project structure should be save,
* general conventions should be save,
* each function should be documented in docstring,
* each function should be covered by unit tests,
* args and return should have typings (for type hints).

### Run unit-tests

Minimal: install `pytest` and `credo_cf` as editable dependency and run in command line:

```bash
pip install pytest
pip install -e .

pytest
```

Recommended: install `tox` via `pip install tox`.
Install all python interpreters listed in `tox.ini` in `envlist = [...]` line. 
Then, run in command line:
 
```bash
tox
```

All test should be passed before make pull request.
