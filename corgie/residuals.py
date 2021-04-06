import torch

def shift_by_int(img, x_shift, y_shift, is_res=False):
    if is_res:
        img = img.permute(0, 3, 1, 2)

    x_shifted = torch.zeros_like(img)
    if x_shift > 0:
        x_shifted[..., x_shift:, :]  = img[..., :-x_shift, :]
    elif x_shift < 0:
        x_shifted[..., :x_shift, :]  = img[..., -x_shift:, :]
    else:
        x_shifted = img.clone()

    result = torch.zeros_like(img)
    if y_shift > 0:
        result[..., y_shift:]  = x_shifted[..., :-y_shift]
    elif y_shift < 0:
        result[..., :y_shift]  = x_shifted[..., -y_shift:]
    else:
        result = x_shifted.clone()

    if is_res:
        result = result.permute(0, 2, 3, 1)

    return result


def res_warp_res(res_a, res_b, is_pix_res=True, permute_field=True):
    if is_pix_res:
        res_b = 2 * res_b / (res_b.shape[-2])

    if len(res_a.shape) == 4:
        result = gridsample_residual(
                        res_a.permute(0, 3, 1, 2),
                        res_b,
                        padding_mode='border').permute(0, 2, 3, 1)
    elif len(res_a.shape) == 3:
        result = gridsample_residual(
                        res_a.permute(2, 0, 1).unsqueeze(0),
                        res_b.unsqueeze(0),
                        padding_mode='border')[0].permute(1, 2, 0)
    else:
        raise Exception("Residual warping requires BxHxWx2 or HxWx2 format.")

    return result


def res_warp_img(img, res_in, is_pix_res=True,
        padding_mode='zeros', mode="bilinear",
        permute_field=True):
    if permute_field:
        res_in = res_in.permute(0, 2, 3, 1)

    if is_pix_res:
        res = 2 * res_in / (img.shape[-1])
    else:
        res = res_in
    if len(img.shape) == 4:
        result = gridsample_residual(img, res, padding_mode=padding_mode, mode=mode)
    elif len(img.shape) == 3:
        if len(res.shape) == 3:
            result = gridsample_residual(img.unsqueeze(0),
                                         res.unsqueeze(0), padding_mode=padding_mode,
                                         mode=mode)[0]
        else:
            img = img.unsqueeze(1)
            result = gridsample_residual(img,
                                         res,
                                         mode=mode,
                                         padding_mode=padding_mode).squeeze(1)
    elif len(img.shape) == 2:
        result = gridsample_residual(img.unsqueeze(0).unsqueeze(0),
                                     res.unsqueeze(0),
                                     padding_mode=padding_mode,
                                     mode=mode)[0, 0]
    else:
        raise Exception("Image warping requires BxCxHxW or CxHxW format." +
                        "Recieved dimensions: {}".format(len(img.shape)))

    return result


def combine_residuals(a, b, is_pix_res=True):
    return b + res_warp_res(a, b, is_pix_res=is_pix_res)


def upsample_residuals(residuals, factor=2.0):
    original_dim = len(residuals.shape)
    while len(residuals.shape) < 4:
        residuals = residuals.unsqueeze(0)
    res_perm = residuals.permute(0, 3, 1, 2)
    result = torch.nn.functional.interpolate(res_perm, scale_factor=factor, mode='bicubic').permute(0, 2, 3, 1)
    result *= factor
    while len(result.shape) > original_dim:
        result = result.squeeze(0)
    return result


def downsample_residuals(residuals):
    original_dim = len(residuals.shape)
    while len(residuals.shape) < 4:
        residuals = residuals.unsqueeze(0)
    result = torch.nn.functional.avg_pool2d(residuals.permute(
                                     0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    result /= 2
    while len(result.shape) > original_dim:
        result = result.squeeze(0)
    return result


def gridsample(source, field, padding_mode, mode='bilinear'):
    """
    A version of the PyTorch grid sampler that uses size-agnostic conventions.
    Vectors with values -1 or +1 point to the actual edges of the images
    (as opposed to the centers of the border pixels as in PyTorch 4.1).
    `source` and `field` should be PyTorch tensors on the same GPU, with
    `source` arranged as a PyTorch image, and `field` as a PyTorch vector
    field.
    `padding_mode` is required because it is a significant consideration.
    It determines the value sampled when a vector is outside the range [-1,1]
    Options are:
     - "zero" : produce the value zero (okay for sampling images with zero as
                background, but potentially problematic for sampling masks and
                terrible for sampling from other vector fields)
     - "border" : produces the value at the nearest inbounds pixel (great for
                  masks and residual fields)
    If sampling a field (ie. `source` is a vector field), best practice is to
    subtract out the identity field from `source` first (if present) to get a
    residual field.
    Then sample it with `padding_mode = "border"`.
    This should behave as if source was extended as a uniform vector field
    beyond each of its boundaries.
    Note that to sample from a field, the source field must be rearranged to
    fit the conventions for image dimensions in PyTorch. This can be done by
    calling `source.permute(0,3,1,2)` before passing to `gridsample()` and
    `result.permute(0,2,3,1)` to restore the result.
    """
    if source.shape[2] != source.shape[3]:
        raise NotImplementedError('Grid sampling from non-square tensors '
                                  'not yet implementd here.')
    scaled_field = field * source.shape[2] / (source.shape[2] - 1)
    return torch.nn.functional.grid_sample(source, 
                                           scaled_field, 
                                           mode=mode,
                                           padding_mode=padding_mode,
                                           align_corners=False)


def gridsample_residual_2d(source, residual, padding_mode):
    """
    Similar to `gridsample()`, but takes a residual field.
    This abstracts away generation of the appropriate identity grid.
    """
    source = torch.FloatTensor(source).unsqueeze(0).unsqueeze(0)
    residual = torch.FloatTensor(residual).unsqueeze(0)
    return gridsample_residual(source, residual, padding_mode)


def gridsample_residual(source, residual, padding_mode, mode="bilinear"):
    """
    Similar to `gridsample()`, but takes a residual field.
    This abstracts away generation of the appropriate identity grid.
    """
    field = residual + identity_grid(residual.shape, device=residual.device)
    return gridsample(source, field, padding_mode, mode=mode)


def _create_identity_grid(size, device='cpu'):
    with torch.no_grad():
        id_theta = torch.FloatTensor([[[1,0,0],[0,1,0]]]).to(device) # identity affine transform
        I = torch.nn.functional.affine_grid(id_theta,torch.Size((1,1,size,size)), align_corners=False)
        I *= (size - 1) / size # rescale the identity provided by PyTorch
        return I


def identity_grid(size, cache=False, device='cpu'):
    """
    Returns a size-agnostic identity field with -1 and +1 pointing to the
    corners of the image (not the centers of the border pixels as in
    PyTorch 4.1).
    Use `cache = True` to cache the identity for faster recall.
    This can speed up recall, but may be a burden on cpu/gpu memory.
    `size` can be either an `int` or a `torch.Size` of the form
    `(N, C, H, W)`. `H` and `W` must be the same (a square tensor).
    `N` and `C` are ignored.
    """
    if isinstance(size, torch.Size):
        if (size[2] == size[3] # image
            or (size[3] == 2 and size[1] == size[2])): # field
            size = size[2]
        else:
            raise ValueError("Bad size: {}. Expected a square tensor size.".format(size))
    if size in identity_grid._identities:
        return identity_grid._identities[size].to(device)
    I = _create_identity_grid(size, device)
    if cache:
        identity_grid._identities[size] = I
    return I.to(device)
identity_grid._identities = {}
