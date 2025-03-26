import os
import gzip
import pickle
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pprint import pp
from warnings import warn

from mcfnmr.config import (
    PROCESSED_UCSFDIR,
    OLDBPEAKSDIR,
    SMARTMINER_DATADIR,
    MM_PEAKLISTDIR,
    MM_LIBDIR,
    MM_MIXLISTSDIR,
    OLDBFULLSPECTRADIR,
    DEBUG, OLDB1DTESTDIR,
)
from mcfnmr.utils.rasterspectrum import RasterSpectrum
from mcfnmr.utils.pointspectrum import PointSpectrum
from mcfnmr.demodata import metaboMinerLibNames, metaboMinerLibNameDict, correctSubstrateNames
from mcfnmr.utils.loading import parseBrukerSpectrum
from copy import deepcopy

aliases = {
    # From MM mixtures
    "Oxoglutarate": "2-Oxoglutarate",
}

userprovided_HMDB_IDs = {
    # From MM mixtures
    "Urea": "294",
    "AMP": "45",  # include
    # From Drosophila annotation
    "Beta-alanine": "56",
    "Glucose-6-P": "1401",
    # From Drosophila annotation
    "DSS": None,  # bmse000795
    # From COLMAR recognitions in N926
    "L-Iditol": 11632,
    "L-Threitol": 2994,
    "Sulfoacetic acid": 258590,
    "1,2_Propanediol": 1881,
    "D-Pinitol": 34219,
    "4-Methyl-2-oxovaleric acid": None,  # bmse000383
    "N-Acetyl-glucosamine 1-phosphate": 1367,
    "D-Glucosaminic acid": 341308,
    "Fructose 1-phosphate": 1076,
    "compound_unknown": None,
    "Isovaleraldehyde": 6478,
    "Neopterin": 845,
    "2-Oxohexane": 5842,
    "Cadaverine": 2322,
    "alpha-Ketoglutaric acid": 208,
    "Adonitol": 508,
    "6-Aminohexanoic acid": 1901,
    "Hydroxyphenylacetylglycine": 735,
    "Gulonic acid": 3290,
    "4-Hydroxyphenylglycine": 244973,
    "Galactonolactone": 2541,
    "Stachyose": 3553,
    "cis-4-Octenedioic acid": 4982,
    "L-Gulonolactone": 3466,
    "D-galactono-1,4-Lactone": 2541,
    "delta-Hexanolactone": 453,
    "D-Xylonate": 59750,
    "scyllo-Inositol": 6088,
    "alpha-D-Glucose-1-phosphate": 1586,
    "meso-Erythritol": 2994,
    "alpha,epsilon-Diaminopimelic acid": 1370,
    "Arbutin": 29943,
    "Argininosuccinic acid": 52,
    "gamma-Aminobutyric acid": 112,
    "Trimethyl phosphate": 259237,
    "N-Acetyl-D-glucosamine-6-phosphate": 1062,
    "Chitosan": 3404,
    "Dehydroascorbic acid": 1264,
    "O-phosphothreonine": 11185,
    "Maltotetraose": 1296,
    "Xylulose": 751,
    "Tetramethylammonium": None,  # bmse000780
    "muco-Inositol": 62138,
    "DL-alpha-Glycerol phosphate": 126,
    "Hexanoic acid": 535,
    "N-Carbamoyl-L-glutamic acid": 15673,
    "Malic acid": 156,
    # From SMART-Miner recognitions in N926
    "beta-Gentiobiose": 341292,
    "Unknown": None,
    "Propylene glycol": 1881,
    "Sodium nalidixic acid": 14917,
    "Nalidixic acid sodium salt": 14917,
    "Muramic acid": 3254,
    "Citicoline": 1413,
    "Allothreonine": 4041,
    "7-Methyladenine": 11614,
    "4-Hydroxyproline": 725,
    "3-Pyridylacetic acid": 1538,
    "3-Methylcrotonaldehyde": 12157,
    "3-Methyl-2-butenal": 12157,
    "3-Methyladenine": 11600,
    "3-Mercaptopyruvic acid": 1368,
    "2-Heptanone": 3671,
    "L-2-Aminobutyric acid": 452,
    "2-Amino-5-ethyl-1,3,4-thiadiazole": None,  # bmse000176
    "1-Methyladenine": 11599,
    "p-Cresol": 1858,
    "Mesaconic acid": 749,
    "L-Homocysteic acid": 2205,
    "Citramalic acid": 426,
    "Arecoline hydrobromide": 30353,
    "4-Heptanone": 4814,
    "4-Aminoantipyrine": 246350,
}


def buildHMDBID2NameMap(verb=1):
    name2id = buildName2HMDBIDMap(verb=verb)
    id2name = dict()
    for name in sorted(name2id.keys()):
        id2name[name2id[name]] = name
    return id2name


def buildName2HMDBIDMap(verb=DEBUG):
    metaboDBFile = MM_LIBDIR / "hsqc.xml"
    et = ET.parse(metaboDBFile)
    libElems = dict([(elem.attrib["Name"], elem) for elem in list(et.getroot())])

    # Use full lib for compound name lookup
    lib = libElems[metaboMinerLibNames[0]]

    # construct mapping name → HMDB ID
    print("Building HMDB-ID list ...")
    map_name2HMDB = {name: str(ID) for name, ID in userprovided_HMDB_IDs.items()}
    for compound in list(lib):
        name = compound.attrib["Name"]
        HMDB_ID = compound.attrib["HMDB_ID"]
        if len(HMDB_ID) == 0:
            warn("No HMDB ID for '%s'!" % name)
            continue
        else:
            HMDB_ID = str(int(HMDB_ID[4:]))
        if name in map_name2HMDB:
            smaller = min(HMDB_ID, map_name2HMDB[name])
            larger = max(HMDB_ID, map_name2HMDB[name])
            if larger != smaller:
                warn(
                    f"\nDuplicate entry for '{name}'! Using smaller ID ({smaller}) and discarding '{larger}'"
                )
            HMDB_ID = smaller
        map_name2HMDB[name] = HMDB_ID
        if name in aliases:
            alias = aliases[name]
        elif name[-3:] == "ate":
            alias = name[:-3] + "ic acid"
        elif name[-7:] == "ic acid":
            alias = name[:-7] + "ate"
        else:
            alias = None
        if verb > 0:
            print(f"   {name} → {HMDB_ID}")

        # Ignore compounds with userprovided_HMDB_IDs[alias]==None
        if alias in userprovided_HMDB_IDs and userprovided_HMDB_IDs[alias] is None:
            print(f"Ignoring alias '{alias}' for {name} with ID '{HMDB_ID}'")
            alias = None

        if alias:
            map_name2HMDB[alias] = HMDB_ID
            if verb > 0:
                print(f"   {alias} → {HMDB_ID}")
    return map_name2HMDB


def loadSMARTMINERBenchmarks():
    # Load Benchmarks from SMART-Miner Publication
    fn = os.path.join(SMARTMINER_DATADIR, "benchmarktable.csv")
    tab = pd.read_csv(fn, index_col=False)
    tab.columns = [col.strip() for col in tab.columns]
    print(tab)

    df = {
        "mixID": [],
        "libID": [],
        "matchType": [],
        "assignmentRadius": [],
        "binning": [],
        "absorptionCost": [],
        "precision": [],
        "recall": [],
        "F1 score": [],
        "cutoffFlag": [],
        "hueID": [],
        "method": [],
        "FP": [],
        "TP": [],
        "FN": [],
    }

    parseMethods = ["Metabominer", "COLMAR-HSQC", "SMART-Miner"]
    n = len(tab["Method"])
    for i in range(n):
        method = tab["Method"][i]
        if method not in parseMethods:
            print(f"Skipping method '{method}'")
            continue
        elif method == "Metabominer":
            # Unify spelling
            method = "MetaboMiner"
        df["mixID"].append(tab["Sample"][i].strip())
        df["libID"].append("Biofluid ( all )")
        df["recall"].append(tab["Recall"][i])
        df["precision"].append(tab["Precision"][i])
        df["F1 score"].append(tab["F-score"][i])
        df["FP"].append(tab["FP"][i])
        df["FN"].append(tab["FN"][i])
        df["TP"].append(tab["TP"][i])
        df["hueID"].append(None)
        df["method"].append(method)

    for k in list(df.keys()):
        if len(df[k]) == 0:
            df[k] = None
    df = pd.DataFrame(df)
    return df


def loadMetaboMinerSpectrum(mixID, denoising_pars={}):
    fn = PROCESSED_UCSFDIR / (mixID + ".pickle")
    spec = parseMetaboMinerDataFile(fn, denoising_pars=denoising_pars)
    return spec


def parseMetaboMinerDataFile(fn, denoising_pars={}):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded data from '{fn}'")
    fullData = data["matrix"]
    header = data["header"]

    if DEBUG > 1:
        print("Header:")
        pp(header)

    spec = RasterSpectrum(
        fullData, header, denoising_pars=denoising_pars, name=fn.name[:-7]
    )
    return spec


def loadMMMixPointSpectra():
    mixPeaks = {}
    for fn in (MM_PEAKLISTDIR).iterdir():
        fn_split = fn.name.split(".")
        specID = ".".join(fn_split[:-1])
        ext = fn_split[-1]
        mixID, specType = specID.split("_")
        if (specType != "hsqc") or ext != "txt":
            continue
        data = np.genfromtxt(fn, dtype=float, skip_header=1)
        coords = data[:, :2]
        weights = data[:, 2]
        spec = PointSpectrum(coords, weights)
        mixPeaks[mixID] = spec
        print(f"Loaded point spectrum for mix '{mixID}' with %d peaks." % len(weights))
    return mixPeaks


def loadMMMixtureLists():
    files = [f for f in (MM_MIXLISTSDIR).iterdir() if f.name[-9:] == "_HMDB.txt"]
    mixLists = {}
    for fn in files:
        mixID = fn.name[:-9]
        with open(fn, "r") as f:
            mixLists[mixID] = [l.strip() for l in f.readlines() if l.strip()]
        print(f"Loaded mix '{mixID}' content list of length %d" % len(mixLists[mixID]))
    return mixLists


def loadMMLibrary(libName="metabominer", verb=False):
    if libName.lower() == "metabominer" or libName is None:
        # Default MM lib is full "Biofluid (all)" library
        libName = metaboMinerLibNames[0]
    elif libName.lower().find("metabominer") > -1:
        libName = metaboMinerLibNameDict["-".join(libName.split("-")[1:]).lower()]
        
    metaboDBFile = MM_LIBDIR / "hsqc.xml"
    et = ET.parse(metaboDBFile)
    libElems = dict([(elem.attrib["Name"], elem) for elem in list(et.getroot())])
    lib = libElems[libName]

    print("Loading MM lib '%s' containing %d compounds..." % (libName, len(lib)))

    peakLists = {}
    for compound in list(lib):
        name = compound.attrib["Name"]

        # Ignore userspecified IDs
        if name in userprovided_HMDB_IDs and userprovided_HMDB_IDs[name] is None:
            assert(False)
            continue

        HMDB_ID = compound.attrib["HMDB_ID"]
        if len(HMDB_ID) == 0:
            warn("No HMDB ID for '%s'!" % name)
        else:
            HMDB_ID = str(int(HMDB_ID[4:]))
        if HMDB_ID in peakLists:
            raise Exception(
                f"\nDuplicate peaklist entry for '{HMDB_ID}' in lib '{libName}'!"
            )
        pls = list(compound)
        if len(pls) > 1:
            warn("   Found %d peaklists for '%s'!? (using first...)" % (len(pls), name))
            raise Exception(f"Several peaklists for '{HMDB_ID}' in lib '{libName}'!")
        elif len(pls) == 0:
            warn("   Found no peaklists for '%s'!?" % (name))
            raise Exception(f"Missing peaklist for '{HMDB_ID}' in lib '{libName}'!")
            # continue
        pl = pls[0]
        if verb:
            print(f"{name} (HMDB-ID {HMDB_ID}):")
            print("   Number of peaks: %d" % len(pl))

        # Coords and weights (latter are not provided by MetaboMiner)
        coords = []
        for peak in pl:
            x, y = peak.attrib["X_PPM"], peak.attrib["Y_PPM"]
            coords.append((float(y), float(x)))
        peakLists[HMDB_ID] = PointSpectrum(
            weights=np.ones(len(coords)), coords=coords, name=name
        )
        if verb:
            print("   peaks:", peakLists[HMDB_ID]["coords"])
    # Manual removal of duplicate entries (which have the same spectrum)
    if ("190" in peakLists) and ("1311" in peakLists):
        # Lactic Acid vs D-Lactic Acid
        print("Removing duplicate entry for D-Lactic Acid (HMDB 1311)")
        del peakLists["1311"]

    return peakLists


def loadOLDBMixSpectrum(specID, smooth, noiseFilterLevel):
    specFiles = sorted(
        [fn for fn in (OLDBFULLSPECTRADIR).iterdir() if fn.name[-7:] == ".txt.gz"]
    )
    specFiles = {"_".join(fn.name.split(".")[0].split("_")[1:]): fn for fn in specFiles}

    pp(specFiles)
    if specID not in specFiles:
        raise Exception(
            f"Couldn't find mixture spectrum '{specID}'. Might need to download it to {OLDBFULLSPECTRADIR}."
        )
    fn = specFiles[specID]
    fullData, header = parseOLDBTextFile(fn)
    spec = RasterSpectrum(
        fullData,
        header,
        name=specID,
        denoising_pars=dict(smoothRadius=smooth, noiseFilterLevel=noiseFilterLevel),
    )
    return spec


def parseOLDBTextFile(fn):
    return parseBrukerSpectrum(fn)


def projectLibTo1D(lib, axis=1):
    compounds = sorted(lib)
    lib1D = dict()
    for c in compounds:
        spec2D = lib[c]
        spec1D = deepcopy(spec2D)
        nc = len(spec2D.coords)
        dim = len(spec1D.coords[0,:])
        for i in range(dim):
            if i == axis:
                continue
            else:
                spec1D.coords[:,i] = np.zeros(nc)
        lib1D[c] = spec1D
    return lib1D


def add_1D_extra_cpds_oldb(lib):
    lib["Water"] = PointSpectrum(coords = np.array([[0.0, 3.33]]), 
                                 weights = np.array([1.0]), 
                                 name = "water")
    lib["DMSO"] = PointSpectrum(coords = np.array([[0.0, 2.54]]), 
                                 weights = np.array([1.0]), 
                                 name = "DMSO")
    lib["TMSP"] = PointSpectrum(coords = np.array([[0.0, 0.0]]), 
                                 weights = np.array([1.0]), 
                                 name = "TMSP")
    

def loadOLDBCompoundLib(project_1D = False, add_extra = False):
    # "add_extra" adds spectra for Water, DMSO, and TMSPD
    # "project_1D" projects spectra to H-axis
    if add_extra and not project_1D:
        warn("loadOLDBCompoundLib() ignoring add_extra (only implemented for 1D spectra)")
        add_extra = False
    peakData = loadOLDBPeakData()
    df = peakData["peaklists_compounds"]
    substances = sorted(set(df["Substance"]))
    lib = {}
    for s in substances:
        # print("s = '%s'"%s)
        dfs = df[df["Substance"] == s]
        weights = []
        coords = []
        for i, f1, f2 in zip(dfs["Integral [abs]"], dfs["ν(F1) [ppm]"], dfs["ν(F2) [ppm]"]):
            # print("   Peak at (%g, %g) with weight: %g"%(f1, f2, i))
            weights.append(i)
            coords.append((f1, f2))
        s = correctSubstrateNames.get(s,s)
        spec = PointSpectrum(coords=coords, weights=weights, name=s)
        lib[s] = spec
    if project_1D:
        # Assuming H-coordinate is second (i.e., axis 1)
        lib = projectLibTo1D(lib, axis=1)
        if add_extra:
            add_1D_extra_cpds_oldb(lib)
    return lib



def bin_1D(weights, coords, binsize):
    n = len(weights)
    dim = coords.shape[1]
    n_binned = int(np.floor(np.round(n/binsize,10)))
    weights_binned = np.zeros(n_binned)
    coords_binned =  np.zeros((n_binned, dim))
    
    for i in range(n_binned):
        i0, i1 = i*binsize, (i+1)*binsize
        weights_binned[i] = np.sum(weights[i0:i1])
        coords_binned[i, 1] = (coords[i0, 1] + coords[i1, 1])/2
    return weights_binned, coords_binned


def loadOLDB_1D_test(target_spec_name, binsize):
    fn = OLDB1DTESTDIR / f"{target_spec_name}.csv.gz"
    with gzip.open(fn, "rb") as f:
        df = pd.read_csv(f)
    H_shifts = np.array(df["chem. Shift/ppm"])
    weights = np.maximum(0.0, np.array(df["rel. intensity"]))
    coords = np.array([np.zeros_like(H_shifts), H_shifts]).T
    weights, coords = bin_1D(weights, coords, binsize)
    spec = PointSpectrum(coords, weights, name=f"1D_test_binsize{binsize}")
    return spec
    

def loadOLDBPeakData():
    # Load all peak data from OL experiments
    data_files = [fn for fn in os.listdir(OLDBPEAKSDIR) if fn[-3:] == "csv"]
    data = {}
    for fn in data_files:
        dataID = fn[:-4]
        print("Loading data '%s' ..." % dataID)
        fullfn = OLDBPEAKSDIR / fn
        data[dataID] = pd.read_csv(fullfn, sep="\t", header=0)
    return data
