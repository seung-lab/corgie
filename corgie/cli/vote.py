import torch
from corgie.log import logger as corgie_logger
from corgie import scheduling, argparsers, helpers, stack

class VoteJob(scheduling.Job):
    def __init__(self,
                 input_fields,
                 output_field,
                 chunk_xy,
                 chunk_z,
                 bcube,
                 mip,
                 softmin_temp,
                 blur_sigma):
        self.input_fields     = input_fields   
        self.output_field     = output_field
        self.chunk_xy         = chunk_xy
        self.chunk_z          = chunk_z
        self.bcube            = bcube
        self.mip              = mip
        self.softmin_temp     = softmin_temp 
        self.blur_sigma       = blur_sigma
        super().__init__()

    def task_generator(self):
        tmp_key = list(self.input_fields.keys())[0]
        tmp_layer = self.input_fields[tmp_key]
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=self.mip)

        tasks = [VoteTask(input_fields=self.input_fields,
                            output_field=self.output_field,
                            mip=self.mip,
                            bcube=chunk,
                            softmin_temp=self.softmin_temp,
                            blur_sigma=self.blur_sigma) for chunk in chunks]

        corgie_logger.debug("Yielding VoteTask for bcube: {}, MIP: {}".format(
                                self.bcube, self.mip))
        yield tasks

class VoteTask(scheduling.Task):
    def __init__(self, 
                 input_fields, 
                 output_field, 
                 mip,
                 bcube,
                 softmin_temp,
                 blur_sigma):
        """Find median vector for each location is set of fields 
        
        Notes: 
            Does not ignore identity fields.
            Padding & cropping is not necessary.

        Args:
            input_fields (Stack): collection of fields
            output_field (Layer)
            mip (int)
            bcbue (BoundingCube)
        """
        super().__init__()
        self.input_fields     = input_fields
        self.output_field     = output_field
        self.mip              = mip
        self.bcube            = bcube
        self.softmin_temp     = softmin_temp 
        self.blur_sigma       = blur_sigma

    def execute(self):
        z = self.bcube.z[0]
        bcube = self.bcube
        mip = self.mip
        fields = [f.read(bcube, mip=mip) for f in self.input_fields.values()]
        fields = torch.cat([f for f in fields]).field()
        voted_field = fields.vote(softmin_temp=self.softmin_temp,
                                  blur_sigma=self.blur_sigma)
        self.output_field.write(data_tens=voted_field, 
                                    bcube=bcube, 
                                    mip=self.mip)