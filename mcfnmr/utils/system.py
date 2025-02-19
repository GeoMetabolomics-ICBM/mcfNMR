import os
from pathlib import Path


def get_mcfnmr_home():
    mcfnmr_home = os.environ.get("MCFNMR_HOME", None)
    if mcfnmr_home is None:
        home_candidate = Path(__file__).absolute().parent.parent.parent
        set_home_candidate = f"export MCFNMR_HOME='{home_candidate}'"
        raise Exception(
            f"Environment variable MCFNMR_HOME not set. Please set it to the path, where mcfnmr has been downloaded to."
            f"\nProbably by: '{set_home_candidate}')."
        )
    mcfnmr_home_abs = Path(mcfnmr_home).absolute()
    if not mcfnmr_home_abs.exists():
        raise mcfnmr_home_abs(
            f"Path '{mcfnmr_home_abs}' (derived from environment variable MCFNMR_HOME='{mcfnmr_home}') doesn't exist."
        )
    subdirs = [str(p.name) for p in mcfnmr_home_abs.iterdir()]
    if not mcfnmr_home_abs.is_dir():
        raise Exception(
            f"Path '{mcfnmr_home_abs}' (derived from environment variable MCFNMR_HOME='{mcfnmr_home}') isn't a directory."
        )
    if (not "tests" in subdirs) or (not "mcfnmr" in subdirs) or (not "data" in subdirs):
        raise Exception(
            f"Path '{mcfnmr_home_abs}' (derived from environment variable MCFNMR_HOME='{mcfnmr_home}') "
            "doesn't contain directories 'tests', 'data', and 'mcfnmr'. Please make sure, that the variable "
            "points to the checkout-directory."
        )
    return mcfnmr_home_abs
