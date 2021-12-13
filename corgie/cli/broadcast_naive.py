from copy import deepcopy
from math import log
from corgie.log import logger as corgie_logger
from corgie import scheduling, helpers
from corgie.stack import PyramidDistanceFieldSet


class BroadcastNaiveJob(scheduling.Job):
    def __init__(
        self,
        block_fields,
        stitching_field,
        output_field,
        chunk_xy,
        bcube,
        pad,
        block_zs,
        mip,
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
        """
        self.block_fields = block_fields
        self.stitching_field = stitching_field
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.pad = pad
        self.block_zs = block_zs
        self.mip = mip
        self.cummulative_stitch_field = self.stitching_field.get_sublayer(
            name='cummulative',
            layer_type='field',
        )
        super().__init__()

    def task_generator(self):
        # Copy the final fields of the first block
        first_block_bcube = self.bcube.copy()
        first_block_bcube.reset_coords(
            zs=self.block_zs[0][0], ze=self.block_zs[0][-1], in_place=True,
        )
        copy_job = CopyLayerJob(
            src_layer=self.block_fields[0],
            dst_layer=self.output_field,
            mip=self.mip,
            bcube=first_block_bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=1,
        )
        yield from copy_job

        # Copy the cumulative field for the first block transition
        first_transition_bcube = self.bcube.copy()
        first_transition_bcube.reset_coords(
            zs=self.block_zs[0][-1], ze=self.block_zs[0][-1] + 1, in_place=True,
        )
        copy_job = CopyLayerJob(
            src_layer=self.stitching_field,
            dst_layer=self.cummulative_stitch_field,
            mip=self.mip,
            bcube=first_transition_bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=1,
        )
        yield from copy_job
        yield scheduling.wait_until_done

        # For each next block
        for i in range(1, len(self.block_zs)):
            this_zs = self.block_zs[i]
            last_zs = self.block_zs[i - 1]
            # Produce final output for the current block
            curr_block_bcube = self.bcube.copy()
            curr_block_bcube.reset_coords(
                zs=this_zs[0], ze=this_zs[-1], in_place=True,
            )
            chunks = self.output_field.break_bcube_into_chunks(
                bcube=curr_block_bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
            )
            tasks  = []
            tasks = [
                ComposeTask(
                    input_fields=[
                        self.cummulative_stitch_field,
                        self.block_fields[i],
                    ],
                    output_field=self.output_field,
                    mip=self.mip,
                    bcube=chunk,
                    pad=self.pad,
                    z_list=[
                        last_zs[-1],
                        this_zs[-1],
                    ],
                ) for chunk in chunks
            ]

            corgie_logger.debug(
                "Yielding Compose for bcube: {}, MIP: {}".format(curr_transition_bcube, self.mip)
            )
            yield tasks

            # Compute the cummulative stitch field for next transition
            curr_transition_bcube = self.bcube.copy()
            curr_transition_bcube.reset_coords(
                zs=this_zs[-1], ze=this_zs[-1] + 1, in_place=True,
            )
            chunks = self.output_field.break_bcube_into_chunks(
                bcube=curr_transition_bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
            )
            tasks  = []
            tasks = [
                ComposeTask(
                    input_fields=[
                        self.cummulative_stitch_field,
                        self.stitching_field,
                    ],
                    output_field=self.cummulative_stitch_field,
                    mip=self.mip,
                    bcube=chunk,
                    pad=self.pad,
                    z_list=[
                        last_zs[-1],
                        this_zs[-1],
                    ],
                ) for chunk in chunks
            ]

            corgie_logger.debug(
                "Yielding Transition Accumulation Compose for bcube: {}, MIP: {}".format(curr_transition_bcube, self.mip)
            )
            yield tasks
            yield scheduling.wait_until_done


class ComposeTask(scheduling.Task):
    def __init__(
        self, input_fields, output_field, mip, bcube, pad, z_list
    ):
        """Compose set of fields,

        Args:
            input_fields ([Layers]): collection of fields
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            pad (int)
            z_list ([int]): list of z indices
        """
        super().__init__()
        self.input_fields = input_fields
        self.output_field = output_field
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.z_list = z_list

    def execute(self):
        corgie_logger.debug(f"ComposeWithDistanceTask, {self.bcube}")
        corgie_logger.debug(f"input_fields: {self.input_fields}")
        corgie_logger.debug(f"z_list: {self.z_list}")
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        fields = FieldSet(
            layers=self.input_fields,
        )
        field = fields.read(bcube=pbcube, z_list=self.z_list, mip=self.mip)
        cropped_field = helpers.crop(field, self.pad)
        self.output_field.write(cropped_field, bcube=self.bcube, mip=self.mip)
