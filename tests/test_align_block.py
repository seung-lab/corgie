from corgie.test_tools import run_test_command

#def test_vanilla_align_block():
#    run_test_command(
#        'align_block_x0',
#    )

#def test_seethrough_align_block():
#    run_test_command(
#        'align_block_st_x0',
#    )

#def test_vv_align_block():
#    run_test_command(
#        'align_block_vv_x0',
#    )

def test_seethrough_vv_align_block():
    run_test_command(
        'align_block_st_vv_x0',
        distributed=True
    )

#def test_seethrough_vv_align_block():
#    run_test_command(
#        'align_block_tiny_x0',
#        distributed=True
#    )
