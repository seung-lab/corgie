class Block:
    def __init__(self, z_start, z_end, special_args={}):
        self.z_start = z_start
        self.z_end = z_end
        self.special_args = special_args

    def __str__(self):
        return f"Block {self.z_start}->{self.z_end}"

    def __repr__(self):
        return self.__str__()
