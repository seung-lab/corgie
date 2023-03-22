import click

from corgie import scheduling
from corgie.log import logger as corgie_logger
from corgie.log import configure_logger


@click.command()
@click.option('--lease_seconds', '-l', nargs=1, type=int, required=True)
@scheduling.scheduler_click_options
@click.option('-v', '--verbose', count=True, help='Turn on debug logging')
def worker(lease_seconds, verbose, **kwargs):
    """Processes corgie tasks from a task queue.
    
    You can run many of these in parallel from
    any machine that can access the queue and
    required data.
    """
    configure_logger(verbose)
    executor = scheduling.parse_executor_from_kwargs(kwargs)
    executor.execute(lease_seconds=lease_seconds)
