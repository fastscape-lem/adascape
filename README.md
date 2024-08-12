[![test build](https://github.com/fastscape-lem/adascape/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/fastscape-lem/adascape/actions)
[![test notebooks](https://github.com/fastscape-lem/adascape/actions/workflows/test_notebooks.yml/badge.svg?branch=master)](https://github.com/fastscape-lem/adascape/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7794374.svg)](https://doi.org/10.5281/zenodo.7794374)



# AdaScape: Adaptive speciation and landscape evolution model

The "AdaScape" package contains a simple adaptive speciation model written
in Python that is coupled with the landscape evolution model [FastScape](https://fastscape.readthedocs.io/en/latest/).

## Install

This package depends on Python (3.9 or later is recommended),
[numpy](http://www.numpy.org/),
[scipy](https://docs.scipy.org/doc/scipy/reference/),
[pandas](https://pandas.pydata.org/),
[fastscape](https://github.com/fastscape-lem/fastscape) and 
[orographic precipitation](https://github.com/fastscape-lem/orographic-precipitation) .

This package also provides a [dendropy](https://dendropy.org/) extension and 
uses [toytree](https://toytree.readthedocs.io/en/latest/index.html) 
to plot phylogenetic trees (optional dependencies).

To install the package locally, first clone this repository:

``` shell
$ git clone https://github.com/fastscape-lem/adascape.git
$ cd adascape
```

Then run the command below (this will install numpy, scipy and pandas
if those libraries are not yet installed in your environment):

``` shell
$ pip install .
```

To install the package for development purpose, use the following
command instead:

``` shell
$ pip install -e .
```

## Usage

Some examples are shown in the ``notebooks`` folder (Jupyter Notebooks).

## Tests

To run the tests, you need to have
[pytest](https://docs.pytest.org/en/latest/) installed in your environment.

Then simply run from this repository's root directory:

``` shell
$ pytest adascape -v
```
