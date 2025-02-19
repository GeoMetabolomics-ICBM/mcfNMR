#! /usr/bin/bash
# Source this here (in MCFNMR_HOME directory) to prepare
# environment for using MCFNMR 

if [ -z ${MCFNMR_HOME+x} ]; then 
	export MCFNMR_HOME="$PWD"
	echo "Set MCFNMR_HOME=$MCFNMR_HOME";
fi

if ! [[ ":$PYTHONPATH:" == *":$MCFNMR_HOME:"* ]]; then
	export PYTHONPATH="$PYTHONPATH":"$MCFNMR_HOME"
	echo "Added MCFNMR_HOME to PYTHONPATH=$PYTHONPATH";	
fi

# Create virtualenv
if ! [ -d "$MCFNMR_HOME"/.mcfnmr_venv ]; then

	echo "Creating virtual environment in $MCFNMR_HOME/.mcfnmr_venv"
	python3 -m virtualenv "$MCFNMR_HOME"/.mcfnmr_venv
	rc=$?
	if [ $rc -gt 0 ]; then
		echo "Command 'python -m virtualenv $MCFNMR_HOME/.mcfnmr_venv' failed with code $rc. Install 'virtualenv' and 'python' first?"
		return $rc;
	fi	
else
	echo "Directory $MCFNMR_HOME/.mcfnmr_venv exists. Assuming virtual environment already exists."
	echo "Use scripts/activate_env.sh to enter."
    return 1;
fi

source "$MCFNMR_HOME"/scripts/activate_env.sh
rc=$?
if [ $rc -gt 0 ]; then
	echo "Activating virtual environment failed."
	return $rc;
fi

# Install/update Python dependencies
python -m pip install -U pip
pip install -r requirements.txt

# Deactivate virtualenv
deactivate

echo "Virtual environment should be ready (if no errors appeared above)."
echo "Use 'source scripts/activate_env.sh to enter.";




