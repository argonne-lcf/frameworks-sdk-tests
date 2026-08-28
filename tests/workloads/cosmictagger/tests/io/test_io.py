import unittest
import pytest
import os, sys
import subprocess


# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
network_dir = network_dir.rstrip("tests/")
sys.path.insert(0,network_dir)


@pytest.mark.io
@pytest.mark.parametrize('synthetic', [False, True])
def test_io(synthetic):
    
    
    # Instead of calling the python objects, use subprocesses 

    # first, where is the exec.py?
    exec_script = network_dir + "/bin/exec.py"

    file_path = network_dir + "/example_data/"
    file_path += "cosmic_tagging_light.h5"

    args = [exec_script, "iotest"]
    args += ["--synthetic", f"{synthetic}"]

    if not synthetic:
        args += ["--file", f"{file_path}"]


    args += ["--iterations",  "10"]



    completed_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)



    if completed_proc.returncode != 0:
        pytest.fail(
            "iotest exited with status {}\nstdout:\n{}\nstderr:\n{}".format(
                completed_proc.returncode,
                completed_proc.stdout.decode(errors="replace"),
                completed_proc.stderr.decode(errors="replace"),
            )
        )

if __name__ == '__main__':
    test_io(synthetic=False)
