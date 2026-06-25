# Media Utils


[![PyPI Status](https://img.shields.io/pypi/v/mediautils.svg)](https://pypi.python.org/pypi/mediautils)
[![Build Status](https://github.com/francois-durand/mediautils/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/francois-durand/mediautils/actions?query=workflow%3Abuild)
[![Documentation Status](https://github.com/francois-durand/mediautils/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/francois-durand/mediautils/actions?query=workflow%3Adocs)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://codecov.io/gh/francois-durand/mediautils/branch/main/graphs/badge.svg)](https://codecov.io/gh/francois-durand/mediautils/tree/main)

A small toolbox of utilities for managing photo and video files.


- Free software: MIT
- Documentation: <https://francois-durand.github.io/mediautils/>.
- Github: <https://github.com/francois-durand/mediautils>


## Features

- Update metadata from standardized file name, and vice-versa.
- Sort photos depending on orientation.

## Quickstart

Install Media Utils:

```console
$ pip install mediautils
```

Use Media Utils in a Python project:

```python
from mediautils import process_standard_files

process_standard_files("in", "out")
```

## Credits

This package was created with [Cookiecutter][CC] and the [Package Helper 3][PH3] project template.

[CC]: <https://github.com/audreyr/cookiecutter>
[PH3]: <https://balouf.github.io/package-helper-3/>
