#! /usr/bin/bash
# Source this here (in MCFNMR_HOME directory) to prepare
# environment for using MCFNMR 

if [ -z ${MCFNMR_HOME+x} ]; then 
	export MCFNMR_HOME="$PWD"
	echo "Set MCFNMR_HOME=$MCFNMR_HOME";
fi

source "$MCFNMR_HOME"/scripts/install.sh
source "$MCFNMR_HOME"/scripts/activate_env.sh
rc=$?
if [ $rc -gt 0 ]; then
	echo "Activating virtual environment failed."
	return $rc;
fi

# Development dependencies
pip install -r "$MCFNMR_HOME"/requirements_tests.txt
rc=$?
if [ $rc -gt 0 ]; then
    echo "Installing requirements for tests/development failed."
    echo "Check pip output for errors.";
    deactivate
    return $rc
else
    echo "Development dependencies installation done."
fi

# Deactivate virtualenv
deactivate

