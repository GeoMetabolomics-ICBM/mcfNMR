#! /usr/bin/python
# Use this script to start the tests.
import os
import subprocess
from pathlib import Path
from mcfnmr.utils.system import get_mcfnmr_home

mcfnmr_home = get_mcfnmr_home()
testdir = Path(mcfnmr_home) / "tests"
env = os.environ.copy()
env["TEXTTEST_HOME"] = str(testdir)

# Run texttest
subprocess.call(["texttest"], env=env)
