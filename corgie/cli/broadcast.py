import torch
from math import log
from corgie.log import logger as corgie_logger
from corgie import scheduling, argparsers, helpers, stack
from corgie.stack import DistanceFieldSet


class BroadcastJob(scheduling.Job):
    def __init__(
        self, input_fields, output_field, chunk_xy, bcube, pad, z_list, mip, decay_dist
    ):
        """
        Args:
            input_fields ([Layers]): collection of fields
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            pad (int)
            z_list ([int]): list of z indices
            decay_dist (float): distance for influence of previous section
        """
        self.input_fields = input_fields
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.pad = pad
        self.z_list = z_list
        self.mip = mip
        self.decay_dist = decay_dist
        super().__init__()

    def task_generator(self):
        tmp_layer = self.input_fields[0]
        chunks = tmp_layer.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
        )

        tasks = []
        for chunk in chunks:
            z_list = self.z_list + [chunk.z_range()[0]]
            task = BroadcastTask(
                input_fields=self.input_fields,
                output_field=self.output_field,
                mip=self.mip,
                bcube=chunk,
                pad=self.pad,
                z_list=z_list,
                decay_dist=self.decay_dist,
            )
            tasks.append(task)

        corgie_logger.debug(
            "Yielding BroadcastTask for bcube: {}, MIP: {}".format(self.bcube, self.mip)
        )
        yield tasks


class BroadcastTask(scheduling.Task):
    def __init__(self, input_fields, output_field, mip, bcube, pad, z_list, decay_dist):
        """Compose set of fields, adjusted by distance

        Order of fields are from target to source, e.g.
            $f_{0 \leftarrow 2} \circ f_{2 \leftarrow 3}$ : [0, 2, 3]

        If len(input_fields) < len(z_list), then repeat input_fields. For
        example, we might only pass a single field, or only the even and odd
        block fields.

        Args:
            input_fields ([Layers]): collection of fields
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            pad (int)
            z_list ([int]): list of z indices
            decay_dist (float): distance for influence of previous section
        """
        super().__init__()
        self.input_fields = input_fields[::-1]
        self.output_field = output_field
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.z_list = z_list
        self.decay_dist = decay_dist
        self.trans_adj = helpers.percentile_trans_adjuster

        if len(z_list) != len(self.input_fields):
            fmul = len(z_list) // len(input_fields)
            frem = len(z_list) % len(input_fields)
            input_fields = input_fields[::-1]
            input_fields = fmul * input_fields + input_fields[:frem]
            input_fields = input_fields[::-1]
        self.input_fields = input_fields

    def execute(self):
        corgie_logger.debug(f"BroadcastTask, {self.bcube}")
        corgie_logger.debug(f"input_fields: {self.input_fields}")
        corgie_logger.debug(f"z_list: {self.z_list}")
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        fields = DistanceFieldSet(decay_dist=self.decay_dist, layers=self.input_fields)
        field = fields.read(bcube=pbcube, z_list=self.z_list, mip=self.mip)
        cropped_field = helpers.crop(field, self.pad)
        self.output_field.write(cropped_field, bcube=self.bcube, mip=self.mip)