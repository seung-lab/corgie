from corgie.test_tools import run_test_command

def test_normalize():
    run_test_command(
        'normalize_x0',
    )

def test_normalize_distributed():
    run_test_command(
        'normalize_x0',
        distributed=True
    )
