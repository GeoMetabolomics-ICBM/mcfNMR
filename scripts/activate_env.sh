#! /usr/bin/bash
# Source this here (in MCFNMR_HOME directory) to prepare
# environment for using MCFNMR 
ENV_NAME=$1
if ! [ $ENV_NAME ]; then
    ENV_NAME=".mcfnmr_venv";
fi

if [ -z ${MCFNMR_HOME+x} ]; then 
	export MCFNMR_HOME="$PWD"
	echo "Set MCFNMR_HOME=$MCFNMR_HOME";
fi

if ! [[ ":$PYTHONPATH:" == *":$MCFNMR_HOME:"* ]]; then
	export PYTHONPATH="$PYTHONPATH":"$MCFNMR_HOME"
	echo "Added MCFNMR_HOME to PYTHONPATH=$PYTHONPATH";	
fi

# Determine operating system
if [[ "$OSTYPE" == "win32"* ]]; then
	OS="WIN"
elif [[ "$OSTYPE" == "cygwin" ]]; then
	# cygwin shell
	OS="WIN"
elif [ "$OSTYPE" == "msys" ]; then
	# MinGW shell
	OS="WIN"
elif [[ "$OSTYPE" == "darwin"* ]]; then
	OS="OSX"
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
	OS="UNIX"
else
	echo "Unknown operating system identifier OSTYPE=$OSTYPE."
	echo "Assuming this to be a unix-derivate and crossing fingers that everything works."
	OS="UNIX"
fi

# activate virtualenv
if [ $OS == "WIN" ]; then
	venv_bindir="$MCFNMR_HOME/$ENV_NAME"/Scripts/
elif [ $OS == "UNIX" ]; then
	venv_bindir="$MCFNMR_HOME/$ENV_NAME"/bin/
elif [ $OS == "OSX" ]; then
	# This is untested...
	venv_bindir="$MCFNMR_HOME/$ENV_NAME"/bin/
fi
source "$venv_bindir"/activate

rc=$?
if [ $rc -gt 0 ]; then
	echo "Activating virtual environment in '$venv_bindir' failed"
	return $rc;
fi
