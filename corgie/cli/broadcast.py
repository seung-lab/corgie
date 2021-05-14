from copy import deepcopy
from math import log
from corgie.log import logger as corgie_logger
from corgie import scheduling, helpers
from corgie.stack import DistanceFieldSet


class BroadcastJob(scheduling.Job):
    def __init__(
        self,
        block_field,
        stitching_fields,
        output_field,
        chunk_xy,
        bcube,
        pad,
        z_list,
        mip,
        decay_dist,
    ):
        """
        Args:
            block_field (Layer): final field
            stitching_fields ([Layers]): collection of fields that will warp in order
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            pad (int)
            z_list ([int]): list of z indices where stitching_fields should be sampled
            decay_dist (float): distance for influence of previous section
        """
        self.block_field = block_field
        self.stitching_fields = stitching_fields
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.pad = pad
        self.z_list = z_list
        self.mip = mip
        self.decay_dist = decay_dist
        super().__init__()

    def task_generator(self):
        chunks = self.output_field.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
        )

        tasks = []
        for chunk in chunks:
            task = BroadcastTask(
                block_field=self.block_field,
                stitching_fields=self.stitching_fields,
                output_field=self.output_field,
                mip=self.mip,
                bcube=chunk,
                pad=self.pad,
                z_list=self.z_list,
                decay_dist=self.decay_dist,
            )
            tasks.append(task)

        corgie_logger.debug(
            "Yielding BroadcastTask for bcube: {}, MIP: {}".format(self.bcube, self.mip)
        )
        yield tasks


class ComposeWithDistanceTask(scheduling.Task):
    def __init__(self, input_fields, output_field, mip, bcube, pad, z_list, decay_dist):
        """Compose set of fields, adjusted by distance

        Order of fields are from target to source, e.g.
            $f_{0 \leftarrow 2} \circ f_{2 \leftarrow 3}$ : [0, 2, 3]

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
        self.input_fields = input_fields
        self.output_field = output_field
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.z_list = z_list
        self.decay_dist = decay_dist
        self.trans_adj = helpers.percentile_trans_adjuster

    def execute(self):
        corgie_logger.debug(f"ComposeWithDistanceTask, {self.bcube}")
        corgie_logger.debug(f"input_fields: {self.input_fields}")
        corgie_logger.debug(f"z_list: {self.z_list}")
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        fields = DistanceFieldSet(decay_dist=self.decay_dist, layers=self.input_fields)
        field = fields.read(bcube=pbcube, z_list=self.z_list, mip=self.mip)
        cropped_field = helpers.crop(field, self.pad)
        self.output_field.write(cropped_field, bcube=self.bcube, mip=self.mip)


class BroadcastTask(ComposeWithDistanceTask):
    def __init__(
        self,
        block_field,
        stitching_fields,
        output_field,
        mip,
        bcube,
        pad,
        z_list,
        decay_dist,
    ):
        """Compose set of stitching_fields, adjusted by distance, with block_field.

        Args:
            block_field (Layer): most recent field, that last to be warped
            stitching_fields ([Layers]): collection of fields at stitching interfaces,
                which are assumed to alternate
            output_field (Layer)
            mip (int)
            bcube (BoundingCube): xy location of where to sample all fields;
                z location of where to sample block_field and where to write composed field
            pad (int)
            z_list ([int]): list of z locations for where to sample each stitching_field;
                length of z_list will indicate how many times to repeat the stitching_fields list
            decay_dist (float): distance for influence of previous section
        """
        corgie_logger.debug(f"z_list: {z_list}")
        corgie_logger.debug(f"stitching_fields: {stitching_fields}")
        input_fields = stitching_fields[::-1]
        if len(z_list) != len(input_fields):
            fmul = len(z_list) // len(input_fields)
            frem = len(z_list) % len(input_fields)
            input_fields = input_fields[::-1]
            input_fields = fmul * input_fields + input_fields[:frem]
            input_fields = input_fields[::-1]
        super().__init__(
            input_fields=input_fields + [block_field],
            output_field=output_field,
            mip=mip,
            pad=pad,
            bcube=bcube,
            z_list=z_list + [bcube.z_range()[0]],
            decay_dist=decay_dist,
        )
