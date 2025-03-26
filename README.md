# mcfNMR - Minimum Cost Flow NMR mixture reconstruction

mcfNMR is a tool for recovering constituent compounds from an NMR spectrum of a mixture sample.
It constructs an optimal approximation ([in terms of the Earth Mover's Distance](https://en.wikipedia.org/wiki/Wasserstein_metric)) of the 
mixture spectrum by combining single compound spectra from a library. For details, see [our publication on mcfNMR](https://pubs.acs.org/doi/10.1021/acs.analchem.4c01652).

![mcfNMR flowchart](data/img/flowchart.svg)

## Install mcfNMR

To install `mcfNMR` for an existing Python environment:

	> pip install mcfNMR

makes the commands ['mcfNMR'](#configuration-file) and ['spec2csv'](#preparing-target-spectra) available.

## Setting up development version mcfNMR

### Pre-requisites

#### Windows

If not installed, download and install python (any version >= 3.10 should work) 
from [python.org](https://www.python.org/downloads/).

If not installed, download and install git from [git-scm.org](https://git-scm.com/downloads). 
Be sure to install the Git Bash (default) or have another version of the bash shell on your system. 
Open bash and `cd` to the directory, which holds downloaded repositories (e.g. $HOME/Workspace).
In the following it is assumed that you use bash for all command line operations.

### Check out the repository

	> git clone https://github.com/GeoMetabolomics-ICBM/mcfNMR <CHECKOUT_DIR>
	> cd <CHECKOUT_DIR>
	> export MCFNMR_HOME="$PWD"

### Installing python environment

For convenience, we provide the script `install.sh` that performs all the steps described below.
This supposes that [virtualenv](https://virtualenv.pypa.io/) is installed. If this is missing from your system, 
you can probably install it by (but see instructions [here](https://virtualenv.pypa.io/en/latest/installation.html))

    > python -m pip --user install virtualenv

__After virtualenv is installed__, to install MCFNMR type

    > source scripts/install.sh
    
This script creates a virtual environment in `<MCFNMR_HOME>/.mcfnmr_venv` and installs the required python packages.

## Running mcfNMR after initial setup

If you have performed the above installation, the next time you use mcfNMR, you can setup the environment by the activate_env.sh script:

    > source scripts/activate_env.sh
    
This should activate the environment `.mcfnmr_venv` (indicated by the prefix to the prompt in your termnal). If working with the development version, you can execute a user-configured classification run by 

    (.mcfnmr_venv)> python -m mcfnmr -c <config_file>
    
If mcfNMR was installed with pip, this script can be started by 

    (.mcfnmr_venv)> mcfNMR -c <config_file>


#### Configuration file

The configuration file (placeholder `<config_file>` above) is a toml-file (see 'data/user\_templates/config\_basic.toml') with the following obligatory fields:

- `lib`: Path to a library-file containing the names, peak-weights, and -coords of the spectra a set of compounds. The path should 
either be absolute or relative to this configuration file. The filename's root is used as the lib's name (e.g. 'lib.csv' → lib_name='lib').
See below for specification of the library-file contents.
- `target`: Path to a target-file containing the name, peak-weights, and -coords of the target-(i.e., mixture-)spectrum. The path should
either be absolute or relative to this configuration file. See below for specification of the target-file contents. Also see 
[Prepare Target Spectrum](#prepare-target-spectrum) below.
- `assignment_radius`: Assignment radius: Single float (if incremental_fit==false) or a sequence of floats (if incremental_fit==true).
- `detection_threshold`:  Detection threshold (assignment > th indicates containment).

Optional fields:

- `isolated_fit`: Boolean flag (true or false) to indicate whether all compounds shall be fitted independently. Otherwise, assigned
weights must jointly respect the maximal target capacity. (default: false)
- `incremental_fit`: Boolean flag (true or false) to indicate whether a multipass incremental fit should be performed. In this case, 
a sequence of assignment radii must be specified at 'assignment_radius'. Otherwise a single-pass optimization is used. (default: false)
- `output_dir`: Output directory for the result files (mcf-results with extension '.pickle', results with extension '.csv' and graphics 
with user-specified format `gfxformat`). (default: MCFNMR_HOME/output/cache)
- `load`: Whether the result should be loaded if it was already computed before. (default: false)
- `name`: Optional name for this configuration. If given, it is used to determine the result filename. Otherwise, the filename is assigned
automatically, using the config parameters.
- `plot`: If true, this plots the matched compounds on top of the target spectrum and saves it. (default: false)
- `show_plot`: If true, this displays the plot using the matplotlib backend. (only effective if `plot=true`, default: false)
- `gfxformat`: This specifies the format of the graphic output by its common extension, e.g., "svg", "png", "eps", "jpg", etc. 
(only effective if `plot=true`, default: "svg")

#### Library- and target-files

Library- and target files must be csv-files, with columns '1H', '13C', and optionally 'weights'. A library file must additionally have a column 
'name' to indicate the compound id the corresponding peaks belong to. See the files 'lib.csv' and 'target.csv' under `data/user_templates/` for examples.
mcfNMR also accepts gzipped files (with extension .csv.gz).

_NOTE_: Although developed with the goal of processing 2D spectra, it is possible to analyze 1D spectra. To represent a 1D-spectrum, simply set the second coordinate to zero (or any fixed, identical value). This should be done for all library and target spectra.


This repository further contains routines to download and prepare demodata, which can then be used for spectral recombination with mcfNMR:
- the HMDB-based compound library, which was provided by the Wishart lab (and used for [MetaboMiner](http://wishart.biology.ualberta.ca/metabominer/)).
- several grid spectra of individual compounds and mixtures provided by GeoMetabolomics @ ICBM, University of Oldenburg ([Dataset link](https://zenodo.org/records/14888536)).
- an automatic extraction of compound spectra from the [BMRB database](https://bmrb.io).

The data can be downloaded with

    > cd $MCFNMR_HOME
    > source scripts/download_demodata.sh [<DATASET>]
    
where the optional argument `<DATASET>` can be one of 'all', 'metabominer', 'oldb', and 'bmrb'. If no argument is given, all data sets are downloaded.
To run the tests, it is not necessary to download the BMRB data. Since the original MetaboMiner website cannot be reached (Feb 19 2025), we download
MetaboMiner files from web.archive.org.


#### Preparing Target Spectra

The mcfnmr package contains a script to convert different data types into a suitable target-csv-file for the main routine.
If mcfnmr was installed with pip, this script can be started by 

    (.mcfnmr_venv)> spec2csv -i input_file [-o output_file -w -b bindims]

Here `input_file` should be one of the supported formats as indicated by its extension. Currently supported: USCF files (.uscf), Bruker matrices (.txt.gz), 
and peak lists (.csv) with columns '1H', '13C', and optionally 'weights' (some basic heuristics try to match these columns to others if not present).
Optionally, an output file can be specified by the option `-o`, otherwise it is set to `input_filename.csv.gz`. The option `-w` indicates that the output file 
should be overridden if it already exists. For grid spectra, the option `-b` allows to specify the size of bins for the target spectrum relative to its original 
resolution. The parameter value (`bindims`) should specify two positive integers, e.g. `-b 2,4` specifies to create a serialized list of coords and intensities
corresponding to 2x4 bins of the original grid.

### Download demodata
We provide a script to download the test-data (OLDB inhouse spectra and MetaboMiner examples) into the folders, 
where the tests expect to find it. The download may take a while since larger amounts of data are downloaded:
	
	> cd $MCFNMR_HOME
	> source scripts/download_demodata.sh 'oldb'
	> source scripts/download_demodata.sh 'metabominer'
	
## Developer Notes

_NOTE_: Development under Windows is not supported, yet. 
		
### Install TextTest, black, and tools to download the demodata

Update pip and setuptools, and install TextTest in the environment:

    > cd $MCFNMR_HOME
    > source scripts/install_dev.sh

### Running the tests

_NOTE 1_: The tests must be executed twice and will only pass successfully in the second run.
This is due to caching operations executed in the first pass.

_NOTE 2_: This happens rarely, but some tests may fail because output of a subprocess running linear
optimization does not synchronize with the output of the main process. We filter the corresponding 
output and unsort its lines, but if a line contains output from both processes, this is not resolved.
Such cases should be easy to identify by looking at the output, though.

To run the tests:

    > cd $MCFNMR_HOME
    > source scripts/run_tests.sh

You should see a window (called the "static gui" of TextTest) with tests in a list on its left area. 
You can mark tests and then click on run to start them. This opens the "dynamic gui", where test progress
and success or failures are reported and can be examined. If running low on memory, try running the 
tests sequentially (check "Run tests sequentially" under tab "Running").

**BEWARE**: the "aggregate\_runs" suite can take _several hours_ to complete for the first run if not 
executed in parallel or on moderately fast CPUs.
    
### Build distribution

The build should be tested in a separate virtual env .build\_env. 
To set this up (if it does not exist, yet) and to rebuild the distribution package, run

    > cd "$MCFNMR_HOME"
    > source scripts/reinstall_distro.sh

This makes the commands ['mcfNMR'](#configuration-file) (an entry point to mcfnmr.main()) and 
['spec2csv'](#preparing-target-spectra) (mcfnmr.utils.spec2csv.main()) available in .build\_env. 
To activate .build\_env:

    > cd "$MCFNMR_HOME"
    > source scripts/activate_env.sh .build_env

The .build\_env environment also has [twine](https://pypi.org/project/twine/), which can be used 
to upload a new mcfNMR version to [PyPI](https://pypi.org/project/mcfNMR) - see 
[packaging guide](https://packaging.python.org/en/latest/guides/section-build-and-publish/).

Before each release, the version number has to be incremented. This should be done using

    > python "$MCFNMR_HOME"/scripts/version_update.py <NEW_VERSION>


