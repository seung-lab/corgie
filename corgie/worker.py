import click

from corgie import scheduling
from corgie.log import logger as corgie_logger
from corgie.log import configure_logger
from corgie.data_backends import set_device

@click.command()
@click.option('--lease_seconds', '-l', nargs=1, type=int, required=True)
@scheduling.scheduler_click_options
@click.option('-v', '--verbose', count=True, help='Turn on debug logging')
@click.option('--device',     '-d', 'device', nargs=1,
                type=str,
                default='cuda',
                help="Pytorch device specification. Eg: 'cpu', 'cuda', 'cuda:0'",
                show_default=True)
def worker(lease_seconds, verbose, device, **kwargs):
    worker_f(lease_seconds, verbose, device, **kwargs)

def worker_f(lease_seconds, verbose, device, **kwargs):
    configure_logger(verbose)
    set_device(device)
    executor = scheduling.parse_executor_from_kwargs(kwargs)
    executor.execute(lease_seconds=lease_seconds)
