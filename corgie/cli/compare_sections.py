import click
import procspec

from corgie import scheduling, argparsers, helpers

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec

from corgie.cli.common import ChunkedJob


class CompareSectionsJob(scheduling.Job):
    def __init__(self, src_stack, dst_layer,
            chunk_xy, processor_spec, mip,
            pad, crop, bcube, tgt_z_offset, tgt_stack=None,
            suffix=''):

        self.src_stack = src_stack
        if tgt_stack is None:
            tgt_stack = src_stack

        self.tgt_stack = tgt_stack
        self.dst_layer = dst_layer
        self.chunk_xy = chunk_xy
        self.pad = pad
        self.crop = crop
        self.bcube = bcube
        self.tgt_z_offset = tgt_z_offset

        self.suffix = suffix

        self.processor_spec = processor_spec
        self.mip = mip
        super().__init__()

    def task_generator(self):
        cs_task = helpers.PartialSpecification(
                CompareSectionsTask,
                processor_spec=self.processor_spec,
                tgt_z_offset=self.tgt_z_offset,
                src_stack=self.src_stack,
                pad=self.pad,
                crop=self.crop,
                tgt_stack=self.tgt_stack,
            )

        chunked_job = ChunkedJob(
                task_class=cs_task,
                dst_layer=self.dst_layer,
                chunk_xy=self.chunk_xy,
                chunk_z=1,
                mip=self.mip,
                bcube=self.bcube,
                suffix=self.suffix
            )

        yield from chunked_job.task_generator


class CompareSectionsTask(scheduling.Task):
    def __init__(self, processor_spec, src_stack, tgt_stack, dst_layer,  mip,
            pad, crop, tgt_z_offset, bcube):
        super().__init__()
        self.processor_spec = processor_spec
        self.src_stack = src_stack
        self.tgt_stack = tgt_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.crop = crop
        self.tgt_z_offset = tgt_z_offset
        self.bcube = bcube

    def execute(self):
        src_bcube = self.bcube.uncrop(self.pad, self.mip)
        tgt_bcube = src_bcube.translate(z_offset=self.tgt_z_offset)

        processor = procspec.parse_proc(
                spec_str=self.processor_spec)

        _, tgt_data_dict = self.tgt_stack.read_data_dict(tgt_bcube,
                mip=self.mip, stack_name='tgt')

        _, src_data_dict = self.src_stack.read_data_dict(src_bcube,
                mip=self.mip, stack_name='src')

        processor_input = {**src_data_dict, **tgt_data_dict}

        result = processor(processor_input, output_key='result')

        cropped_result = helpers.crop(result, self.crop)
        self.dst_layer.write(cropped_result, bcube=self.bcube, mip=self.mip)
