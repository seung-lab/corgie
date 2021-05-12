from corgie.test_tools import run_test_command

def test_upsample_img():
    run_test_command(
        'upsample_img_x0',
        '/usr/people/popovych/corgie/tests/test_data/original/original_x0'
    )

def test_upsample_mask():
    run_test_command(
        'upsample_mask_x0',
        '/usr/people/popovych/corgie/tests/test_data/original/original_x0'
    )
