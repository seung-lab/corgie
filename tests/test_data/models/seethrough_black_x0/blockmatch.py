import torch

import numpy as np
import itertools
import copy


def get_peak(x, eps=1.E-3, randomize=False):
    d = x.shape[-1]
    peak = torch.zeros((x.shape[0], 2), device=x.device, dtype=torch.long)

    for b in range(x.shape[0]):
        m = x[b].argmax()
        initial_peak = [int(m // d), int(m % d)]

        if randomize:
            peak_value = x[b, initial_peak[0], initial_peak[1]].squeeze(0)
            peaks_mask = x[b] >= (peak_value - eps)
            peaks_coords = peaks_mask.nonzero()
            random_peak_id = np.random.randint(0, peaks_coords.shape[0])
            chosen_one = peaks_coords[random_peak_id]
            peak[b, 0] = chosen_one[0]
            peak[b, 1] = chosen_one[1]
        else:
            peak[b, 0] = initial_peak[0]
            peak[b, 1] = initial_peak[1]
    return peak


def zone_out_around_peak(x, peak, zone_dist, fill=None):
    if fill == None:
        fill = float("-inf")
    d = x.shape[-1]
    zoned_x = x.clone()
    for b in range(x.shape[0]):
        x_low = max(peak[b, 0] - zone_dist, 0)
        x_high = min(peak[b, 0] + zone_dist + 1, d)
        y_low = max(peak[b, 1] - zone_dist, 0)
        y_high = min(peak[b, 1] + zone_dist + 1, d)
        zoned_x[b, x_low:x_high, y_low:y_high] = fill
    return zoned_x


def get_two_peaks(x_in, zone_dist=1, eps=1.E-3):
    # x_in is shape BxXxY
    first_peak = get_peak(x_in)
    zoned_x = zone_out_around_peak(x_in, first_peak, zone_dist)
    second_peak = get_peak(zoned_x)
    return first_peak, second_peak


def get_black_mask(img, black_threshold):
    if black_threshold == 0:
        black_mask = img == 0
    else:
        black_mask = img <= black_threshold
    return black_mask


def get_black_fraction(img, black_threshold):
    img_black_px = get_black_mask(img, black_threshold)
    img_black_px_count = torch.sum(img_black_px)
    img_px_count = torch.sum(torch.ones_like(img))
    img_black_fraction = float(img_black_px_count) / float(img_px_count)

    return img_black_fraction


def normalize(img, per_feature_center=True, per_feature_var=False, eps=1e-5,
        mask=None, mask_fill=None):
    img_out = img.clone()

    if mask is not None:
        assert mask.shape == img.shape
    for i in range(1):
        for b in range(img.shape[0]):
            x = img_out[b]
            if per_feature_center and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    if mask is not None:
                        m = mask[b, f]
                        x[f][m] = x[f][m].clone() - torch.mean(x[f][m].clone())
                    else:
                        x[f] = x[f].clone() - torch.mean(x[f].clone())
            else:
                if mask is not None:
                    m = mask[b]
                    x[m] = x[m].clone() - torch.mean(x[m].clone())
                else:
                    x[...] = x.clone() - torch.mean(x.clone())

            if per_feature_var and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    if mask is not None:
                        m = mask[b, f]
                        var = torch.var(x[f][m].clone())
                        x[f][m] = x[f][m].clone() / (torch.sqrt(var) + eps)
                    else:
                        var = torch.var(x[f].clone())
                        x[f] = x[f].clone() / (torch.sqrt(var) + eps)
            else:
                if mask is not None:
                    m = mask[b]
                    var = torch.var(x[m].clone())
                    x[m] = x[m].clone() / (torch.sqrt(var) + eps)
                else:
                    var = torch.var(x.clone())
                    x[...] = x.clone() / (torch.sqrt(var) + eps)

    if mask is not None and mask_fill is not None:
        img_out[mask == False] = mask_fill

    return img_out


def get_index_neighbors(index, shape, diagonals=False, reach=1):
    result = []

    if diagonals:
        offsets = itertools.product(np.arange(0, reach + 2) - 1, repeat=2)
    else:
        offsets = []
        offsets += itertools.product(np.arange(0, reach + 2) - 1, [0])
        offsets += itertools.product([0], np.arange(0, reach + 2) - 1)

    for o in offsets:
        new_index = list(copy.copy(index))
        new_index[0] += o[0]
        new_index[1] += o[1]
        if new_index[0] >= 0 and new_index[1] >=0 and \
           new_index[0] < shape[0] and new_index[1] < shape[1]:
               result.append(new_index)
    return result


def get_neighbor_average(index, field):
    neighbors = get_index_neighbors(index, field.shape)
    result = 0
    valid_count = 0
    nonzero_count = 0
    invalid_count = 0

    for i in neighbors:
        value = field[i[0], i[1]]
        if not np.isnan(value).any() and not np.isinf(value).any():
            valid_count += 1
            if value[0] != 0 or value[1] != 0:
                result += value
                nonzero_count += 1
        else:
            invalid_count += 1

    if valid_count == 0 or (invalid_count > 0 and nonzero_count == 0):
        return None
    elif nonzero_count > 0:
        return result / nonzero_count
    else:
        return 0


def extrapolate_field_missing_values(field):
    field[..., 0] = extrapolate_missing_values(field[..., 0])
    field[..., 1] = extrapolate_missing_values(field[..., 1])
    return field


def extrapolate_missing_values(array):
    missing_map = (np.isnan(array) + np.isinf(array)) > 0
    missing_indexes = np.nonzero(missing_map)
    missing_coords = list(set(zip(missing_indexes[0], missing_indexes[1])))

    missing_count = len(missing_coords)
    while missing_count > 0:
        print ("Purging ", len(missing_coords))
        for i in missing_coords:
            v = get_neighbor_average(i, array)
            if v is not None:
                array[i[0], i[1]] = v

        missing_map = (np.isnan(array) + np.isinf(array)) > 0
        missing_indexes = np.nonzero(missing_map)
        missing_coords = list(set(zip(missing_indexes[0], missing_indexes[1])))
        prev_missing_count = missing_count
        missing_count = len(missing_coords)
        if missing_count >= prev_missing_count:
            print ("Purge stuck, 0 everything")
            break
    array[missing_indexes] = 0


    return array


def block_match(tgt, src, tile_size=16, tile_step=16, max_disp=10, min_overlap_px=500,
                filler="inf", r_delta=1.1):
    src = src.squeeze()
    tgt = tgt.squeeze()

    tile_alignment_pad = (tile_size - tile_step) // 2

    padded_tgt = torch.nn.ZeroPad2d(tile_alignment_pad)(tgt)
    padded_src = torch.nn.ZeroPad2d(tile_alignment_pad)(src)

    max_disp_pad = max_disp
    padded_tgt = torch.nn.ZeroPad2d(max_disp_pad)(tgt)
    padded_src = torch.nn.ZeroPad2d(max_disp_pad)(src)

    img_size = padded_tgt.shape[-1]
    tile_count = 1 + (img_size - max_disp*2 - tile_size) // tile_step
    result = np.zeros((tile_count, tile_count, 2))
    peaks = []
    peak_vals = []
    peak_ratios = []
    for x_tile in range(0, tile_count):
        for y_tile in range(0, tile_count):
            src_tile_coord, tgt_tile_coord = compute_tile_coords(x_tile, y_tile, tile_size,
                                                              tile_step, max_disp, img_size,
                                                                x_offset=max_disp,
                                                                y_offset=max_disp)
            src_tile = padded_src[src_tile_coord]
            tgt_tile = padded_tgt[tgt_tile_coord]
            if get_black_fraction(src_tile, 0) > 0.98 or get_black_fraction(tgt_tile, 0) > 0.98:
                match_displacement = [0, 0]
                if get_black_fraction(src_tile, 0) != 1.0 and \
                        get_black_fraction(tgt_tile, 0) != 1.0:
                    pass
            else:
                ncc = get_ncc(tgt_tile, src_tile, div_by_overlap=True,
                        min_overlap_count=min_overlap_px)
                ncc_np = ncc.squeeze().cpu().numpy()

                if ncc.var() < 1E-13 or ((ncc != ncc).sum() > 0):
                    match_displacement = [float(filler), float(filler)]
                else:
                    peak1, peak2 = get_two_peaks(ncc, 8)
                    peaks.append([peak1, peak2])
                    match = np.unravel_index(ncc_np.argmax(), ncc_np.shape)
                    peak_vals.append([ncc_np[peak1[0][0],  peak1[0][1]],
                                      ncc_np[peak2[0][0],  peak2[0][1]],
                                    ])
                    peak_ratios.append(peak_vals[-1][0] / (peak_vals[-1][1] + 1e-5))
                    if peak_ratios[-1] < r_delta:
                        match_displacement = [float(filler), float(filler)]
                    else:
                        match_tile_start = (tgt_tile_coord[0].start + match[0], tgt_tile_coord[1].start + match[1])
                        src_tile_start   = (src_tile_coord[0].start, src_tile_coord[1].start)
                        match_displacement = np.subtract(src_tile_start, match_tile_start)

            result[x_tile, y_tile, 0] = -match_displacement[1]
            result[x_tile, y_tile, 1] = -match_displacement[0]

    patched_result = extrapolate_missing_values(result)

    result_var = torch.FloatTensor(patched_result).to(src.device).unsqueeze(0)

    scale = tgt.shape[-2] / result_var.shape[-2]
    result_ups_var = torch.nn.functional.interpolate(result_var.permute(0, 3, 1, 2),
            scale_factor=scale, mode='bicubic')

    final_result_var = result_ups_var

    final_result = final_result_var.permute(0, 2, 3, 1)
    return final_result


def filter_black_field(field, img, black_threshold=0):
    black_mask = (img.abs() < black_threshold).squeeze()
    field[..., black_mask] = 0
    return field


def get_ncc(tgt, tmpl, div_by_overlap=False, min_overlap_ratio=0.6,
            min_overlap_count=500):
    tgt = tgt.unsqueeze(0).unsqueeze(0)
    tmpl = tmpl.unsqueeze(0).unsqueeze(0)

    mask_val = 0.05
    tgt_mask = tgt.abs() > mask_val
    tmpl_mask = tmpl.abs() > mask_val

    tgt_norm = normalize(tgt, mask=tgt_mask)
    tmpl_norm = normalize(tmpl, mask=tmpl_mask)
    ncc = get_cc(tgt_norm, tmpl_norm)

    if div_by_overlap:
        overlap_count = get_cc(tgt_mask.float(), tmpl_mask.float(), div_by_tmpl_size=False)
        overlap_ratio = overlap_count / (tmpl_mask != 0).sum()
        adjusted_ncc = ncc / (overlap_count + 1e-5)
        ncc[overlap_count != 0] = adjusted_ncc[overlap_count != 0]
        ncc[overlap_count < min_overlap_count] = ncc.min()
        ncc[overlap_ratio < min_overlap_ratio] = ncc.min()

    return ncc


def get_cc(target, template, feature_weights=None, normalize_cc=False, div_by_tmpl_size=True):
    cc_side = target.shape[-1] - template.shape[-1] + 1
    cc = torch.zeros((target.shape[0], cc_side, cc_side),
            device=target.device, dtype=torch.float)
    for b in range(target.shape[0]):
        cc[b:b+1] = torch.nn.functional.conv2d(target[b:b+1], template[b:b+1]).squeeze()
    if normalize_cc:
        cc = normalize(cc)
    elif div_by_tmpl_size:
        cc = cc / torch.sum(torch.ones_like(template[0], device=template.device))
    return cc


def get_displaced_tile(disp, tile):
    result = copy.deepcopy(tile)
    result[0].start += disp[0]


def compute_tile_coords(x_tile, y_tile, tile_size, tile_step, max_disp, img_size,
                       x_offset=0, y_offset=0):
    src_xs = x_tile * tile_step + x_offset
    src_xe = src_xs + tile_size
    src_ys = y_tile * tile_step + y_offset
    src_ye = src_ys + tile_size

    tgt_xs = max(0, src_xs - max_disp)
    tgt_xe = min(img_size, src_xe + max_disp)
    tgt_ys = max(0, src_ys - max_disp)
    tgt_ye = min(img_size, src_ye + max_disp)

    src_coords = (slice(src_xs, src_xe), slice(src_ys, src_ye))
    tgt_coords = (slice(tgt_xs, tgt_xe), slice(tgt_ys, tgt_ye))

    return src_coords, tgt_coords


def get_patch_middle(coords):
    x_m = (coords[0].start - coords[0].end) / 2
    y_m = (coords[1].start - coords[1].end) / 2

    return(x_m, y_m)
