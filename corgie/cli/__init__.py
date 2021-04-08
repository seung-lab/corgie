from copy import deepcopy

COMMAND_LIST = []

from corgie.cli.downsample import downsample

COMMAND_LIST.append(downsample)
from corgie.cli.upsample import upsample

COMMAND_LIST.append(upsample)
from corgie.cli.compute_field import compute_field

COMMAND_LIST.append(compute_field)
from corgie.cli.compute_stats import compute_stats

COMMAND_LIST.append(compute_stats)
from corgie.cli.normalize import normalize

COMMAND_LIST.append(normalize)
from corgie.cli.align_block import align_block

COMMAND_LIST.append(align_block)
from corgie.cli.align import align

COMMAND_LIST.append(align)
from corgie.cli.render import render

COMMAND_LIST.append(render)
from corgie.cli.copy import copy

COMMAND_LIST.append(copy)
from corgie.cli.apply_processor import apply_processor

COMMAND_LIST.append(apply_processor)
from corgie.cli.invert_field import invert_field

COMMAND_LIST.append(invert_field)
from corgie.cli.create_skeletons import create_skeletons

COMMAND_LIST.append(create_skeletons)
from corgie.cli.transform_skeletons import transform_skeletons

COMMAND_LIST.append(transform_skeletons)
from corgie.cli.filter_skeletons import filter_skeletons

COMMAND_LIST.append(filter_skeletons)
from corgie.cli.merge_copy import merge_copy

COMMAND_LIST.append(merge_copy)
from corgie.cli.compute_field_by_spec import compute_field_by_spec

COMMAND_LIST.append(compute_field_by_spec)
from corgie.cli.merge_render import merge_render

COMMAND_LIST.append(merge_render)
from corgie.cli.apply_processor_by_spec import apply_processor_by_spec

COMMAND_LIST.append(apply_processor_by_spec)
from corgie.cli.normalize_by_spec import normalize_by_spec

COMMAND_LIST.append(normalize_by_spec)
from corgie.cli.downsample_by_spec import downsample_by_spec

COMMAND_LIST.append(downsample_by_spec)
from corgie.cli.compare_sections import compare_sections

COMMAND_LIST.append(compare_sections)

# To add new commands, create a file in this folder implementing a command,
# import the command here and add it to the list:


def get_command_list():
    return COMMAND_LIST
