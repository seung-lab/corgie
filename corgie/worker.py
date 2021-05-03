import click

from corgie import scheduling
from corgie.log import logger as corgie_logger
from corgie.log import configure_logger
from corgie import settings

@click.command()
@click.option('--lease_seconds', '-l', nargs=1, type=int, required=True)
@scheduling.scheduler_click_options
@click.option('-v', '--verbose', count=True, help='Turn on debug logging')
def worker(lease_seconds, verbose, **kwargs):
    configure_logger(verbose)
    settings.IS_WORKER = True
    executor = scheduling.parse_executor_from_kwargs(kwargs)
    executor.execute(lease_seconds=lease_seconds)
