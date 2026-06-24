# Installation

## Stable release

To install Media Utils, run this command in your terminal:

```console
$ pip install mediautils
```

This is the preferred method to install Media Utils, as it will always install the most recent stable release.

If you don't have [pip] installed, this [Python installation guide] can guide
you through the process.

````{note}
If you want to use Media Utils as a dependency in a UV-managed project, add it with
```console
$ uv add mediautils
```
````

## From sources

The sources for Media Utils can be downloaded from the [Github repo].

You can either clone the public repository:

```console
$ git clone git://github.com/francois-durand/mediautils
```

Or download the [tarball]:

```console
$ curl -OJL https://github.com/francois-durand/mediautils/tarball/main
```

Once you have a copy of the source, you can install it from the package directory with:

```console
$ pip install .
```

[github repo]: https://github.com/francois-durand/mediautils
[pip]: https://pip.pypa.io
[python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
[tarball]: https://github.com/francois-durand/mediautils/tarball/main
