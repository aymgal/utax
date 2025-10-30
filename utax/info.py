"""PACKAGE INFO

This module provides some basic information about the package.

"""

# Set the package release version
version_info = (0, 0, 2)
__version__ = '.'.join(str(c) for c in version_info)

# Set the package details
__author__ = 'Aymeric Galan, Austin Peel, Martin Millon, Frederic Dux, Kevin Michalewicz'
__email__ = 'aymeric.galan@gmail.com'
__year__ = '2022'
__url__ = 'https://github.com/aymgal/utax'
__description__ = 'Utility functions for signal processing, compatible with the differentable programming library JAX.'
__python__ = '>=3.10'
__requires__ = [
    'jax>=0.5.0', 
    'jaxlib>=0.5.0', 
]  # Package dependencies

# Default package properties
__license__ = 'MIT'
__about__ = ('{} Author: {}, Email: {}, Year: {}, {}'
             ''.format(__name__, __author__, __email__, __year__,
                       __description__))
__setup_requires__ = ['pytest-runner', ]
__tests_require__ = ['pytest', 'pytest-cov', 'pytest-pep8']
