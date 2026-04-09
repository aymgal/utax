from importlib.metadata import version, PackageNotFoundError

__author__ = 'Aymeric Galan, Austin Peel, Martin Millon, Frederic Dux, Kevin Michalewicz'
__email__ = 'aymeric.galan@gmail.com'

try:
    __version__ = version("utax")
except PackageNotFoundError:
    __version__ = "unknown"
