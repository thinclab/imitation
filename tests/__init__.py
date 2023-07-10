from importlib import metadata

try:
    __version__ = metadata.version("tests")
except metadata.PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass