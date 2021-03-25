import click

from corgie.data_backends.base import get_data_backends, str_to_backend

# when adding a new backend type, include it here
from corgie.data_backends.cvbackend import CVDataBackend
from corgie.data_backends.json import JSONDataBackend
from corgie.data_backends.base import DataBackendBase

# also update default backend if you want to
DEFAULT_DATA_BACKEND = 'cv'

pass_data_backend = click.make_pass_decorator(DataBackendBase)


