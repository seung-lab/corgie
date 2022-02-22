# To add new commands, create a file in this folder implementing a command,
# import the command here and add it to the list:
from corgie.cli.downsample import downsample
from corgie.cli.upsample import upsample
from corgie.cli.compute_field import compute_field
from corgie.cli.compute_stats import compute_stats
from corgie.cli.normalize import normalize
from corgie.cli.align_block import align_block
from corgie.cli.align import align
from corgie.cli.render import render
from corgie.cli.copy import copy
from corgie.cli.apply_processor import apply_processor
from corgie.cli.invert_field import invert_field
from corgie.cli.create_skeletons import create_skeletons
from corgie.cli.transform_skeletons import transform_skeletons
from corgie.cli.filter_skeletons import filter_skeletons
from corgie.cli.merge_copy import merge_copy
from corgie.cli.compute_field_by_spec import compute_field_by_spec
from corgie.cli.merge_render import merge_render
from corgie.cli.apply_processor_by_spec import apply_processor_by_spec
from corgie.cli.normalize_by_spec import normalize_by_spec
from corgie.cli.downsample_by_spec import downsample_by_spec
from corgie.cli.compare_sections import compare_sections
from corgie.cli.seethrough import seethrough_block
from corgie.cli.fill_nearest import fill_nearest
from corgie.cli.vote import vote
from corgie.cli.combine_masks import combine_masks
from corgie.cli.create_segmentation_masks import create_segmentation_masks
from corgie.cli.multi_section_compare import multi_section_compare

COMMAND_LIST = []
COMMAND_LIST.append(downsample)
COMMAND_LIST.append(upsample)
COMMAND_LIST.append(compute_field)
COMMAND_LIST.append(compute_stats)
COMMAND_LIST.append(compare_sections)
COMMAND_LIST.append(normalize)
COMMAND_LIST.append(align_block)
COMMAND_LIST.append(align)
COMMAND_LIST.append(render)
COMMAND_LIST.append(copy)
COMMAND_LIST.append(apply_processor)
COMMAND_LIST.append(invert_field)
COMMAND_LIST.append(create_skeletons)
COMMAND_LIST.append(transform_skeletons)
COMMAND_LIST.append(filter_skeletons)
COMMAND_LIST.append(merge_copy)
COMMAND_LIST.append(compute_field_by_spec)
COMMAND_LIST.append(merge_render)
COMMAND_LIST.append(apply_processor_by_spec)
COMMAND_LIST.append(normalize_by_spec)
COMMAND_LIST.append(downsample_by_spec)
COMMAND_LIST.append(seethrough_block)
COMMAND_LIST.append(fill_nearest)
COMMAND_LIST.append(vote)
COMMAND_LIST.append(compare_sections)
COMMAND_LIST.append(combine_masks)
COMMAND_LIST.append(create_segmentation_masks)
COMMAND_LIST.append(multi_section_compare)


def get_command_list():
    return COMMAND_LIST
