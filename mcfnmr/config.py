import os
import numpy as np
from mcfnmr.utils.system import get_mcfnmr_home

# Global debug-level in {0,1,2} controls verbosity
# and runtime sanity checks
DEBUG = 1

if DEBUG > 0:
    print(
        "config: Enforcing numpy legacy print mode to ensure that numpy floats are printed without type information."
    )
np.set_printoptions(legacy="1.21")

# mcfNMR repository url
REPO_URL = "https://github.com/GeoMetabolomics-ICBM/mcfNMR"

MCFNMR_HOME = get_mcfnmr_home()
OUTDIR = MCFNMR_HOME / "output"
TEMPDIR = MCFNMR_HOME / "tmp"
DATADIR = MCFNMR_HOME / "data"

OLDBDATADIR = DATADIR / "OLDBdata"
OLDBPEAKSDIR = OLDBDATADIR / "peaks"
OLDB1DTESTDIR = OLDBDATADIR / "test1D"
OLDBFULLSPECTRADIR = OLDBDATADIR / "fullspectra"

METABOMINER_DIR = DATADIR / "MetaboMiner"
UCSFDIR = METABOMINER_DIR / "ucsf"
PROCESSED_UCSFDIR = METABOMINER_DIR / "processedUCSF"
SPARKYDIR = METABOMINER_DIR / "sparky"
MM_LIBDIR = METABOMINER_DIR / "metabominer" / "lib"
MM_MIXLISTSDIR = METABOMINER_DIR / "mixture_contents"
MM_PEAKLISTDIR = METABOMINER_DIR / "examples" / "peaklist"
UCSFDATA_BIN = SPARKYDIR / "bin" / "ucsfdata"

SMARTMINER_DATADIR = DATADIR / "SMART-Miner"

COLMAR_DATADIR = DATADIR / "COLMAR"

# Maximal absorption cost (assumed larger than any possibly encountered point distance)
MAX_ABSORPTON_COST = 1e6

# Default scales for HSQC spectra
DEFAULT_SCALE_C = 10.0
DEFAULT_SCALE_H = 1.0

if not OUTDIR.exists():
    os.makedirs(OUTDIR)
    print(f"Created directory '{OUTDIR}'")
if not TEMPDIR.exists():
    os.makedirs(TEMPDIR)
    print(f"Created directory '{TEMPDIR}'")

# This defines default parameters for MCF recombinations
default_pars = dict(
    assignmentRadius=0.01,
    absorptionCost=None,
    regionDefs=None,
    gridInfoY=None,
    # Preprocessing step for full raster spectrum
    smoothing=0,
    # peak resolution of raster to point conversion
    binning=(2, 2),
    # Whether to use the peak list dat of the target (or the raster data)
    # (Only effective if both are available)
    pointTarget=True,
    # Whether to do a simulataneous isolated fit for all compounds
    isolated_fit=False,
    # threshold (relative to noiseStd) for cutting off raster spectrum signal
    noiseFilterLevel=None,
    distPars={"scalex": DEFAULT_SCALE_C, "scaley": DEFAULT_SCALE_H},
)


# Column name candidates for the different dimensions
# (case-insensitive
csv_column_name_links = {
    "weight": ["w", "weights", "weight", "integral", "i", "intensity"],
    "1H": ["1h", "ppm h"],
    "13C": ["13c", "ppm c"],
}
