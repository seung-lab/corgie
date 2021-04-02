import torch
from math import log
from corgie.log import logger as corgie_logger
from corgie import scheduling, argparsers, helpers, stack

class BroadcastJob(scheduling.Job):
    def __init__(self,
                 input_fields,
                 output_field,
                 chunk_xy,
                 bcube,
                 pad,
                 z_list,
                 mip,
                 decay_dist):
        self.input_fields     = input_fields   
        self.output_field     = output_field
        self.chunk_xy         = chunk_xy
        self.bcube            = bcube
        self.pad              = pad
        self.z_list           = z_list
        self.mip              = mip
        self.decay_dist       = decay_dist
        super().__init__()

    def task_generator(self):
        tmp_key = list(self.input_fields.keys())[0]
        tmp_layer = self.input_fields[tmp_key]
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=1,
                    mip=self.mip)

        tasks = [BroadcastTask(input_fields=self.input_fields,
                            output_field=self.output_field,
                            mip=self.mip,
                            bcube=chunk,
                            pad=self.pad,
                            z_list=self.z_list, 
                            decay_dist=self.decay_dist) for chunk in chunks]

        corgie_logger.debug("Yielding BroadcastTask for bcube: {}, MIP: {}".format(
                                self.bcube, self.mip))
        yield tasks

class BroadcastTask(scheduling.Task):
    def __init__(self, 
                 input_fields, 
                 output_field, 
                 mip,
                 bcube,
                 pad,
                 z_list,
                 decay_dist):
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
        self.input_fields     = input_fields
        self.output_field     = output_field
        self.mip              = mip
        self.pad              = pad
        self.bcube            = bcube
        self.z_list           = z_list
        self.decay_dist       = decay_dist
        self.trans_adj        = helpers.percentile_trans_adjuster

        field_multiple = len(self.z_list) // len(self.input_fields)
        if field_multiple > 1:
            self.input_fields = field_multiple * self.input_fields

    def adjust_bcube(self, bcube, z):
        return bcube.reset_coords(zs=z, ze=z+1, inplace=False)

    def get_field(self, layer, bcube, mip, dist):
        """Get field, adjusted by distance

        Args:
            layer (Layer)
            bcube (BoundingCube)
            mip (int)
            dist (float)
        
        Returns:
            TorchField adjusted (blurred/attenuated) by distance
        """
        # TODO: add ability to get blurred field using trilinear interpolation
        c = min(max(dist / self.decay_dist, 0.), 1.)
        f = layer.read(bcube=bcube, mip=mip).field_()
        return f * c

    def execute(self):
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        src_z = self.z_list[-1]
        z = self.z_list[0]
        layer = self.input_fields[0]
        abcube = self.adjust_bcube(pbcube, z)
        agg_field = self.get_field(layer=layer,
                                    bcube=abcube,
                                    mip=self.mip,
                                    dist=src_z - z)
        for z, layer in zip(self.z_list[1:], self.input_fields[1:]):
            trans = helpers.percentile_trans_adjuster(agg_field)
            trans.x = (trans.x // (2**self.mip)) * 2**self.mip
            trans.y = (trans.y // (2**self.mip)) * 2**self.mip
            abcube = self.adjust_bcube(pbcube, z)
            abcube = abcube.translate(x_offset=int(trans.x),
                                      y_offset=int(trans.y))
            agg_field -= trans
            agg_field = agg_field.from_pixels()
            this_field = self.get_field(layer=layer,
                                        bcube=abcube,
                                        mip=self.mip,
                                        dist=src_z - z)
            this_field = this_field.from_pixels()
            agg_field = agg_field(this_field)
            agg_field = agg_field.pixels()
            agg_field += trans
        cropped_field = helpers.crop(agg_field, self.pad)
        self.output_field.write(cropped_field, bcube=self.bcube, mip=self.mip)