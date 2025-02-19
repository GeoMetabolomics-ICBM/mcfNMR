#! /usr/bin/bash

source "$MCFNMR_HOME"/scripts/activate_env.sh
export TEXTTEST_HOME="$MCFNMR_HOME"/tests/

python "$MCFNMR_HOME"/tests/run_tests.py
deactivate

