import sys
import os
import yaml

from click.testing import CliRunner
from corgie.main import cli

from corgie.test_tools import run_command

my_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    runner = CliRunner()
    spec_path = sys.argv[1]

    test_name = os.path.basename(spec_path)
    assert test_name.endswith('.yaml')
    test_name = test_name[:-len('.yaml')]

    dst_path = os.path.join(my_path, 'test_data/gt', test_name)
    result = run_command(test_name, dst_path)
    print (result.output)

    assert result.exit_code == 0

