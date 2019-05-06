# paraspec: Parapatric Speciation Model

The "paraspec" package is a simple Parapatric Speciation Model written
in Python.

## Install

This package depends on Python (3.5 or later is recommended),
[numpy](http://www.numpy.org/),
[scipy](https://docs.scipy.org/doc/scipy/reference/) and
[pandas](https://pandas.pydata.org/).

To install the package locally, first clone this repository:

``` shell
$ git clone https://gitext.gfz-potsdam.de/sec55-public/parapatric-speciation
$ cd parapatric-speciation
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
$ pytest paraspec -v
```
