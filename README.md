# RVFitter

# Installation

Clone the repository and install the package in development mode:

```
pip install -e .
```

# Idea of the project

The project holds all the bookkeeping for normalizing and clipping astronomical spectra. By default it does not contain functionality but rather provides the building blocks to more elaborate astronomical code.

On purpose we did not create a large framwork which only fits our purposes but rather empower more users to tailer the package and its functionality to their needs.

An example how the package can be used can be found in:
```
https://github.com/MaclaRamirez/RVFitter_scripts
```

# TODOs and feature requests

## Short term

* unittests for fitting
* apply the fitting range (wlc-window) to the spectra in velocity space
* add Docstrings


## Medium term
* unify units with the help of astropy.units
  --> allow for setting the unit of the respective nspecs-files

* this implies more configurability in a yaml-config file
  --> maybe it does not but it for sure requires an input-parameter in the creation of the RVFitter-object and it needs to be propagated down to all other classes
