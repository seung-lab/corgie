import torch
from math import log
from corgie.log import logger as corgie_logger
from corgie import scheduling, argparsers, helpers, stack


def compute_softmin_temp(dist, weight, size):
    """Compute softmin temp given binary assumptions

    Assumes that voting subsets are either correct or incorrect.

    Args:
        dist (float): distance between the average differences of
            a correct and incorrect subset.
        weight (float): desired weight to be assigned for correct/incorrect
            distance.
        size (int): size of a subset in voting

    Returns:
        float for softmin temperature that will achieve this weight
    """
    return -dist / (log(1 - weight) - log(weight * size))


class VoteOverFieldsJob(scheduling.Job):
    def __init__(
        self,
        input_fields,
        output_field,
        chunk_xy,
        bcube,
        mip,
        softmin_temp=None,
        blur_sigma=1.0,
    ):
        self.input_fields = input_fields
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.mip = mip
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma
        super().__init__()

    def task_generator(self):
        tmp_key = list(self.input_fields.keys())[0]
        tmp_layer = self.input_fields[tmp_key]
        chunks = tmp_layer.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
        )

        tasks = [
            VoteOverFieldsTask(
                input_fields=self.input_fields,
                output_field=self.output_field,
                mip=self.mip,
                bcube=chunk,
                softmin_temp=self.softmin_temp,
                blur_sigma=self.blur_sigma,
            )
            for chunk in chunks
        ]

        corgie_logger.debug(
            "Yielding VoteTask for bcube: {}, MIP: {}".format(self.bcube, self.mip)
        )
        yield tasks


class VoteOverFieldsTask(scheduling.Task):
    def __init__(
        self, input_fields, output_field, mip, bcube, softmin_temp, blur_sigma=1.0
    ):
        """Find median vector for single location over set of fields

        Notes:
            Does not ignore identity fields.
            Padding & cropping is not necessary.

        Args:
            input_fields (Stack): collection of fields
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
        """
        super().__init__()
        self.input_fields = input_fields
        self.output_field = output_field
        self.mip = mip
        self.bcube = bcube
        if softmin_temp is None:
            softmin_temp = compute_softmin_temp(
                dist=1, weight=0.99, size=len(input_fields)
            )
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma

    def execute(self):
        bcube = self.bcube
        mip = self.mip
        fields = [f.read(bcube, mip=mip) for f in self.input_fields.values()]
        fields = torch.cat([f for f in fields]).field()
        voted_field = fields.vote(
            softmin_temp=self.softmin_temp, blur_sigma=self.blur_sigma
        )
        self.output_field.write(data_tens=voted_field, bcube=bcube, mip=self.mip)


class VoteOverZJob(scheduling.Job):
    def __init__(
        self,
        input_field,
        output_field,
        chunk_xy,
        bcube,
        z_list,
        mip,
        softmin_temp=None,
        blur_sigma=1.0,
    ):
        self.input_field = input_field
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.z_list = z_list
        self.mip = mip
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma
        super().__init__()

    def task_generator(self):
        chunks = self.output_field.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
        )

        tasks = [
            VoteOverZTask(
                input_field=self.input_field,
                output_field=self.output_field,
                mip=self.mip,
                bcube=chunk,
                z_list=self.z_list,
                softmin_temp=self.softmin_temp,
                blur_sigma=self.blur_sigma,
            )
            for chunk in chunks
        ]

        corgie_logger.debug(
            "Yielding VoteOverZTask for bcube: {}, MIP: {}".format(self.bcube, self.mip)
        )
        yield tasks


class VoteOverZTask(scheduling.Task):
    def __init__(
        self,
        input_field,
        output_field,
        mip,
        bcube,
        z_list,
        softmin_temp,
        blur_sigma=1.0,
    ):
        """Find median vector for set of locations in a single field

        Notes:
            Does not ignore identity fields.
            Padding & cropping is not necessary.

        Args:
            input_field (Layer)
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            z_list (range, [int])
        """
        super().__init__()
        self.input_field = input_field
        self.output_field = output_field
        self.mip = mip
        self.bcube = bcube
        self.z_list = z_list
        if softmin_temp is None:
            softmin_temp = compute_softmin_temp(dist=1, weight=0.99, size=len(z_list))
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma

    def execute(self):
        fields = []
        for z in self.z_list:
            bcube = self.bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            fields.append(self.input_field.read(bcube, mip=self.mip))
        fields = torch.cat([f for f in fields]).field()
        voted_field = fields.vote(
            softmin_temp=self.softmin_temp, blur_sigma=self.blur_sigma
        )
        self.output_field.write(data_tens=voted_field, bcube=self.bcube, mip=self.mip)
