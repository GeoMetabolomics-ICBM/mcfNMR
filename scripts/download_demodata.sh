#! /usr/bin/bash
# This downloads test data

WDIR="$PWD"

if [ $# -ge 1 ]; then
    DATASET=$1;
else
    DATASET="all";
fi

echo "$DATASET"

if ! [[ "$DATASET" =~ ^(all|metabominer|oldb|bmrb)$ ]]; then
    echo "dataset (download_demodata.sh argument) must be one of 'all', 'metabominer', 'oldb', and 'bmrb', given: '$DATASET'"
fi

# activate virtualenv
source "$MCFNMR_HOME"/.mcfnmr_venv/bin/activate
rc=$?
if [ $rc -gt 0 ]; then
	echo "Activating virtual environment failed."
	return $rc;
fi

# Install data download dependencies
pip install wget waybackpack nmrpystar

if [ $DATASET = "all" ] || [ $DATASET = "metabominer" ]; then
    # Download and extract MetaboMiner data
    python -m mcfnmr.demodata.metabominer
fi

if [ $DATASET = "all" ] || [ $DATASET = "oldb" ]; then
    # Download OLDB inhouse data
    python -m mcfnmr.demodata.oldbdata
fi

if [ $DATASET = "all" ] || [ $DATASET = "bmrb" ]; then
    # Download BMRB compound library
    python -m mcfnmr.demodata.bmrb
fi

cd "$WDIR"
