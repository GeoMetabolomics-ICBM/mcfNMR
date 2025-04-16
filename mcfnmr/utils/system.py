import os
from pathlib import Path

HOMEDIR_WARNINGS = 0

def get_mcfnmr_home():
    global HOMEDIR_WARNINGS
    mcfnmr_home = os.environ.get("MCFNMR_HOME", None)
    if mcfnmr_home is None:
        # home_candidate = Path(__file__).absolute().parent.parent.parent # only for dev
        # set_home_candidate = f"export MCFNMR_HOME='{home_candidate}'"
        home_candidate = (Path(os.environ["HOME"])/".mcfnmr").absolute()
        if home_candidate.exists():
            if not home_candidate.is_dir():
                raise Exception(
                    f"Environment variable MCFNMR_HOME not set."
                )
            else:
                if HOMEDIR_WARNINGS == 0:
                    print(f"Using MCOM_HOME='{home_candidate}'.")
                    HOMEDIR_WARNINGS += 1
                return home_candidate
        else:
            answer = ""
            answercount, maxcount = 0, 3
            while answer.lower() not in {"y", "yes", "n", "no"} and answercount < maxcount:
                answer = input(f"Environment variable MCFNMR_HOME not set. Use directory '{home_candidate}'? (yes/no)")
            if answer.lower() not in {"y", "yes"}:
                raise Exception(
                    f"Environment variable MCFNMR_HOME not set."
                )
            else:
                os.makedirs(home_candidate)
                print(f"Created directory '{home_candidate}'.")
                print(f"Using MCOM_HOME='{home_candidate}'.")
                return home_candidate
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
