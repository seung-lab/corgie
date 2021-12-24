from dataclasses import astuple, dataclass

import numpy as np
import torch
from copy import deepcopy


class Binarizer:
    def __init__(self, binarization):
        self.bin = binarization

    def __call__(self, tens):
        if self.bin is None:
            return tens
        elif self.bin[0] == "neq":
            return tens != self.bin[1]
        elif self.bin[0] == "eq":
            return tens == self.bin[1]
        elif self.bin[0] == "gt":
            return tens > self.bin[1]
        elif self.bin[0] == "lt":
            return tens < self.bin[1]


class PartialSpecification:
    def __init__(self, f, **kwargs):
        self.f = f
        self.constr_kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, **kwargs):
        return self.f(**self.constr_kwargs, **kwargs)

    def __getitem__(self, k):
        return self.constr_kwargs[k]


@dataclass
class Translation:
    x: float
    y: float

    def __iter__(self):
        return iter(astuple(self))

    def __add__(self, T):
        return Translation(self.x + T.x, self.y + T.y)

    def __sub__(self, T):
        return Translation(self.x - T.x, self.y - T.y)

    def __mul__(self, scalar):
        return Translation(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __floordiv__(self, scalar):
        return Translation(self.x // scalar, self.y // scalar)

    def __truediv__(self, scalar):
        return Translation(self.x / scalar, self.y / scalar)

    def to_tensor(self, **kwargs):
        return torch.tensor([[[[self.x]], [[self.y]]]], **kwargs)

    def round(self, ndigits=None):
        return Translation(round(self.x, ndigits), round(self.y, ndigits))

    def copy(self):
        return deepcopy(self)

    def round_to_mip(self, src_mip, tgt_mip):
        if tgt_mip is None:
            return self.copy()
        elif tgt_mip <= src_mip:
            return Translation(self.x, self.y)  # Return copy
        else:
            snap_factor = 2 ** (tgt_mip - src_mip)
            return (self // snap_factor) * snap_factor


def percentile_trans_adjuster(field, h=25, l=75, unaligned_img=None):
    if field is None:
        result = Translation(0, 0)
    else:
        nonzero_field_mask = (field[:, 0] != 0) & (field[:, 1] != 0)

        if unaligned_img is not None:
            no_tissue = field.field().from_pixels()(unaligned_img) == 0
            nonzero_field_mask[..., no_tissue.squeeze()] = False

        nonzero_field = field[..., nonzero_field_mask.squeeze()].squeeze()

        if nonzero_field.sum() == 0:
            result = Translation(0, 0)
        else:
            low_l = percentile(nonzero_field, l)
            high_l = percentile(nonzero_field, h)
            mid = 0.5 * (low_l + high_l)
            result = Translation(x=int(mid[0]), y=int(mid[1]))

    return result


def percentile(field, q):
    # https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
    :param field: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(0.01 * float(q) * (field.shape[1] - 1))
    result = field.kthvalue(k, dim=1).values
    return result


def crop(**kwargs):
    raise NotImplementedError


def expand_to_dims(tens, dims):
    tens_dims = len(tens.shape)
    assert (tens_dims) <= dims
    tens = tens[(None,) * (dims - tens_dims)]
    return tens


def cast_tensor_type(tens, dtype):
    """
    tens: pytorch tens
    dtype: string, eg 'float', 'int', 'byte'
    """
    if dtype is not None:
        assert hasattr(tens, dtype)
        return getattr(tens, dtype)()
    else:
        return tens


def read_mask_list(mask_list, bcube, mip):
    result = None

    for m in mask_list:
        this_data = m.read(bcube=bcube, mip=mip).to(torch.bool)
        if result is None:
            result = this_data
        else:
            result = result | this_data

    return result


def crop(data, c):
    if c == 0:
        return data
    else:
        if data.shape[-1] == data.shape[-2]:
            return data[..., c:-c, c:-c]
        elif data.shape[-2] == data.shape[-3] and data.shape[-1] == 2:  # field
            return data[..., c:-c, c:-c, :]


def coarsen_mask(mask, n=1, flip=False):
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    kernel_var = (
        torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(mask.device).float()
    )
    k = torch.nn.Parameter(data=kernel_var, requires_grad=False)
    for _ in range(n):
        if flip:
            mask = mask.logical_not()
        mask = torch.nn.functional.conv2d(mask.float(), kernel_var, padding=1) > 1
        if flip:
            mask = mask.logical_not()
        mask = mask

    return mask


def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


class BoolFn:
    def __init__(self, exp):
        """Create a boolean expression from dict describing a linear threshold (LT) neuron

        A LT neuron is described as dict with keys "inputs" and "threshold". The LT neuron
        will evaluate to true if the sum of "inputs" is greater than or equal to the
        "threshold". The "inputs" of an LT neuron maybe include other LT neurons. Final
        variables must be expressed as a dict that does not contain the key "inputs". These
        variables are stored as BoolVars. To evaluate the Boolean expression, pass a method
        that can evaluate the dict for each BoolVar.

        Args:
            exp (dict of dicts): dict describing nested LT neurons. For example:
            ```
            {
            "inputs": [
                {
                "inputs": [
                    { "weight": 1, "key": "a", "offset": 0 },
                    { "weight": -1, "key": "b", "offset": -1 },
                    { "weight": 1, "key": "c", "offset": 2 }
                ],
                "threshold": 2
                },
                {
                "inputs": [
                    { "weight": 1, "key": "a", "offset": 1 },
                    { "weight": 1, "key": "c", "offset": 0 }
                ],
                "threshold": 1
                }
            ],
            "threshold": 0
            }
            ```

            evaluates as
            `((a[0] - b[-1] + c[2] > 2) + (a[1] - c[0]) > 1) > 0`.
        """
        self.inputs = [
            BoolFn(e) if "inputs" in e else BoolVar(e) for e in exp["inputs"]
        ]
        self.threshold = exp["threshold"]

    def __call__(self, fn):
        return sum([i(fn) for i in self.inputs]) > self.threshold


class BoolVar:
    def __init__(self, exp):
        """See description of BoolFn"""
        self.exp = exp

    def __call__(self, fn):
        return fn(self.exp)