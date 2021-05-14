import torch
import os
import pathlib
import copy

import metroem

class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        this_folder = pathlib.Path(__file__).parent.absolute()
        model_folder = os.path.join(this_folder, 'model')
        self.aligner = metroem.aligner.Aligner(model_folder=model_folder, **kwargs)

    def forward(self, **kwargs):
        return self.aligner.forward(**kwargs)

    def save_state_dict(self, checkpoint_folder):
        self.aligner.save_state_dict(checkpoint_folder)


def create(**kwargs):
    return Model(**kwargs)
