import json
import click

from corgie.log import configure_logger
from corgie.log import logger as corgie_logger

from corgie.argparsers import corgie_option, corgie_optgroup
from corgie.data_backends import get_data_backends, str_to_backend, \
        DEFAULT_DATA_BACKEND, DataBackendBase
from corgie.cli.downsample import downsample

from corgie.cli import get_command_list

from corgie import scheduling

class GroupWithCommandOptions(click.Group):
    """ Allow application of options to group with multi command """

    def add_command(self, cmd, name=None):
        click.Group.add_command(self, cmd, name=name)
        # add the group parameters to the command
        for param in self.params:
            cmd.params.append(param)

        # hook the commands invoke with our own
        cmd.invoke = self.build_command_invoke(cmd.invoke)
        self.invoke_without_command = True

    def build_command_invoke(self, original_invoke):

        def command_invoke(ctx):
            """ insert invocation of group function """

            # separate the group parameters
            ctx.obj = dict(_params=dict())
            for param in list(self.params):
                name = param.name
                ctx.obj['_params'][name] = ctx.params[name]
                del ctx.params[name]

            # call the group function with its parameters
            params = ctx.params
            ctx.params = ctx.obj['_params']
            self.invoke(ctx)
            ctx.params = params

            # now call the original invoke (the command)
            original_invoke(ctx)

        return command_invoke


@click.group(cls=GroupWithCommandOptions)
@scheduling.scheduler_click_options
@click.option('--device',     '-d', 'device', nargs=1,
                type=str,
                default='cpu',
                help="Pytorch device specification. Eg: 'cpu', 'cuda', 'cuda:0'",
                show_default=True)
@click.option('-v', '--verbose', count=True, help='Turn on debug logging')
@click.pass_context
def cli(ctx, device, verbose, **kwargs):
    """Align large 3D electron microscopy image stacks.

    The core idea is to take a roughly aligned stack, find
    voxel correspondances using the SEAMLeSS neural network
    technique and perform smooth elastic deformation except
    at image defects where non-smooth transforms are allowed.

    Use the separate `corgie-worker` command to perform
    distributed computation.

    You can read more in the following papers:

    1. Popovych, S., Macrina, T., Kemnitz, N., Castro, M., 
    Nehoran, B., Jia, Z., Bae, J.A., Mitchell, E., Mu, 
    S., Trautman, E.T., Saalfeld, S., Li, K., Seung, S., 
    BioArXiv, 2022. Petascale pipeline for precise 
    alignment of images from serial section electron 
    microscopy. 
    https://doi.org/10.1101/2022.03.25.485816

    2. Mitchell, E., Keselj, S., Popovych, S., Buniatyan, D., 
    Seung, H.S., arXiv 2019. Siamese Encoding and Alignment by 
    Multiscale Learning with Self-Supervision.
    http://arxiv.org/abs/1904.02643

    BARK BARK! Corgie is licensed under Apache 2.0.
    https://github.com/seung-lab/corgie/blob/main/LICENSE
    """
    # This little hack let's us make group options look like
    # child command options, and at the same time only execute
    # the setup once
    if ctx.invoked_subcommand is None:
        configure_logger(verbose)
        ctx.obj = {}
        DataBackendBase.default_device = device
        corgie_logger.debug("Creating scheduler...")
        kwargs['command_name'] = ctx.command.name
        ctx.obj['scheduler'] = scheduling.parse_scheduler_from_kwargs(
                kwargs)

        corgie_logger.debug("Scheduler created.")

for c in get_command_list():
    cli.add_command(c)

if __name__ == "__main__":
    cli()

