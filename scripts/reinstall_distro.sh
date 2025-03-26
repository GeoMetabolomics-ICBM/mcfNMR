#! /usr/bin/bash
# Source this here (in MCFNMR_HOME directory) to prepare
# environment for using MCFNMR 
wdir="$PWD"

if [ -z ${MCFNMR_HOME+x} ]; then
    echo ""
	echo "Please set environment variable MCFNMR_HOME first."
    echo ""
    return 1;
fi

if ! [ -d "$MCFNMR_HOME/.build_env" ]; then
    echo "Virtual environment $MCFNMR_HOME/.build_env doesn't exist. Try creating it." 
    virtualenv "$MCFNMR_HOME/.build_env"
    rc=$?;
else
    rc=0
fi
if [ $rc -gt 0 ]; then
    echo ""
	echo "Creating virtual environment failed."
	return $rc;
fi

cd "$MCFNMR_HOME"
source "$MCFNMR_HOME"/scripts/activate_env.sh .build_env
rc=$?
if [ $rc -gt 0 ]; then
    echo ""
	echo "Activating virtual environment failed."
    cd "$wdir"
	return $rc;
fi

# Remove previous build
cd "$wdir"
if [ -d "$MCFNMR_HOME"/dist ]; then
    rm "$MCFNMR_HOME"/dist -r;
fi

# Ensure, we have build and twine
pip install build twine

# Build
python -m build

# Re-install
pip install --force-reinstall "$MCFNMR_HOME"/dist/*.whl
rc=$?

if [ $rc -gt 0 ]; then
    echo ""
    echo "Reinstallation of mcfNMR failed. Please check ouput for errors."
    return $rc;
else
    echo ""
    echo "Reinstallation done."
    echo "";
fi

