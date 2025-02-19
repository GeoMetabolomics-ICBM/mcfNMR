import os
import urllib.request as rq
from urllib.error import HTTPError
import multiprocessing as mp

import wget
import bs4
import nmrpystar
import numpy as np
import pandas as pd
    
from mcfnmr.config import DATADIR, DEBUG
from pprint import pp
import sys
from argparse import Namespace
import pickle
from mcfnmr.utils.system import get_mcfnmr_home
from mcfnmr.utils.pointspectrum import PointSpectrum
from numpy import vstack
from mcfnmr.utils.spec2csv import saveLibAsCSV

BMRB_DIR = DATADIR / "BMRB"
# This site lists the directory containing .str files for all compounds 
BMRB_FTP_URL = "https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_lists/experimental"

# Destination for str-file download
DOWNLOAD_DIR = BMRB_DIR / "downloads"

HSQC_expnames = [
    # Pre-3.1.1.92
    '2D [1H,13C]-HSQC',
    '2D [1H,13C]-HSQC SW small',
    '2D [1H,13C]-HSQC SW expanded',
    # Post-3.1.1.92
    '2D 1H-13C HSQC',
    '2D 1H-13C HSQC SW small',
    '2D 1H-13C HSQC SW expanded',
    '2D 1H-13C HSQC-TOCSY-ADIA',
    ]
        
known_versions = [
    "NMR STAR v3.1",
    '3.1.0.29',
    "3.1.1.21",
    "3.1.1.31",
    "3.1.1.7",
    "3.1.1.92",
    "3.2.0.16",
    "3.2.1.32",
    "3.2.6.0",
    ]


# HSQC data keys in order of preference
HSQCKeys = [
    "spectral_peak_HSQC", 
    "spectral_peak_1H_13C_HSQC", 
    "spectral_peaks_2D_1H_13C_HSQC_set01",
    "spectral_peaks_2D_1H-13C_HSQC_set01",
    ]

# Keys for the shift table in preference order
# shiftTableKeys = ['Assigned_peak_chem_shift', # This contains multplicities
shiftTableKeys = ['Peak_char',
                  'Spectral_transition_char']
chemShiftField = {'Assigned_peak_chem_shift' : 'Val', 
                  'Peak_char' : 'Chem_shift_val', 
                  'Spectral_transition_char' : 'Chem_shift_val'}
peakIDField = {'Assigned_peak_chem_shift' : 'Peak_ID', 
                  'Peak_char' : 'Peak_ID', 
                  'Spectral_transition_char' : 'Spectral_transition_ID'}

def download_str_file(str_file, dest, download_existent):
        if (dest / str_file).exists() and not download_existent:
            # Skipping download"
            return -1, str_file
        url = "/".join((BMRB_FTP_URL, str_file))
        print(f"Downloading {url} ...")
        try:
            wget.download(url, str(dest / str_file))
            success = True
        except HTTPError as e:
            print(f"Failed to download {str_file}. HTTPError: {e.code}")
            success = False
        return success, str_file


def downloadBMRB(download_existent=False):
    # Download destination for .str files 
    if not DOWNLOAD_DIR.exists():
        os.makedirs(DOWNLOAD_DIR)
        print(f"Created directory {DOWNLOAD_DIR}")
           
    # Use https download
    print(f"Using https connection to download BMRB entries to {DOWNLOAD_DIR} ...")
    
    # Grab site (BMRB_FTP_URL), which lists all relevant .str files
    # and compile a list of their urls
    response = rq.urlopen(BMRB_FTP_URL)
    if response.status != 200:
        print("response.status != 200 (bad url?)")
    if response.msg != "OK":
        print("response.msg != 'OK' (bad url?)")
    doc = "".join(l.decode() for l in response.readlines())
    soup = bs4.BeautifulSoup(doc, 'html.parser')
    str_files = sorted(set(t["href"] for t in soup.find_all("a") if t["href"][-4:]==".str"))
    nstr = len(str_files)   
    print(f"Found {nstr} nmr-star files on BMRB-server.\n\nStarting downloads...\n")
    
    # Download all str-files in parallel
    pool = mp.Pool()
    args = zip(str_files, (DOWNLOAD_DIR for i in range(nstr)), (download_existent for i in range(nstr)))
    results = list(pool.starmap(download_str_file, args))
    failed = [str_file for success, str_file in results if success==False]
    skipped = [str_file for success, str_file in results if success==-1]
    print(f"\nDownloading completed (skipped existing: {len(skipped)}/{nstr}, failed: {len(failed)}/{nstr-len(skipped)}).")


def loopToDataFrame(nmrstar_loop):
    ncols = len(nmrstar_loop.keys)
    coldata = [[r[i] for r in nmrstar_loop.rows] for i in range(ncols)]
    df_exps = dict({c:col for c, col in zip(nmrstar_loop.keys, coldata)})
    return pd.DataFrame(df_exps)


def guessUnknownSpecKey(data):
    candidateHSQCKeys = set()
    # Look for other keys containing HSQC
    for k in data.keys():
        if k.lower().find("hsqc") != -1:
            candidateHSQCKeys.add(k)
    if candidateHSQCKeys:
        print("  Found potential keys: %s"%candidateHSQCKeys)
        print("  Looking for peaklist candidate...")
        good_candis = []
        for k in candidateHSQCKeys:
            candidateSpec = data.get(k, None)
            if candidateSpec is None:
                continue
            hasLoops = getattr(candidateSpec, "loops", None) is not None
            hasShiftTable = False
            if hasLoops:
                for stk in shiftTableKeys:
                    hasShiftTable = hasShiftTable or stk in candidateSpec.loops
            else:
                # Debug
                print("No loops at '%s'"%k)
            goodCandi = hasLoops and hasShiftTable
            if goodCandi:
                good_candis.append(k)
        
        print(f"  Found {len(good_candis)} candidate keys:\n    {good_candis}")
        specKey = good_candis[0]
        print("  I'll try data key '%s'."%k)
    else:
        # Debug
        print("No potential HSQC keys...")
        # print("  Data keys:")
        # pp(list(data.keys()))
        specKey = None
    return specKey


def findSpecKey(data):    
    specKeys = []
    for k in HSQCKeys:
        spec = data.get(k, None)
        if spec is not None:
            specKeys.append(k)
    if specKeys:
        specKey = specKeys[0]
        print("  Using data key '%s'"%k)
    else:
        specKey = None
    return specKey


def reportExperimentList(data):
    if "experiment_list" not in data:
        print("   No entry 'experiment_list' in save frames.")
        HSQC_exp_available = False
        explist = None
    else:
        explist = data["experiment_list"].loops["Experiment"]
        
        df_exps = loopToDataFrame(explist)
        print("Experiment list:")
        pp([n for n in df_exps["Name"]])
        HSQC_exp_available = np.any([n in df_exps["Name"] for n in HSQC_expnames])
        
        # Debug
        if DEBUG:
            newkeys = [n not in HSQC_expnames and n.lower().find("hsqc") != -1 for n in df_exps["Name"]]
            if np.any(newkeys):
                print("print new HSQC key?")
                print(df_exps["Name"][newkeys])
                sys.exit()
            
    # # Assigned_chemical_shifts holds experiments as well
    # chem_shifts = data['assigned_chemical_shifts'].loops
    # exps = loopToDataFrame(chem_shifts["Chem_shift_experiment"])
    
    return HSQC_exp_available


def addBasicEntryInformation(p, entry):
    entry_info = p.value.saves["entry_information"].datums
    entry.name = p.value.name
    entry.title = entry_info["Title"].strip()
    entry.version = entry_info["NMR_STAR_version"]
    entry.orig_version = entry_info.get("Original_NMR_STAR_version", None)


def addCompoundInformation(data, entry):
    chem_comp_1 = data["chem_comp_1"]

    entry.smiles = None
    smilesTable = chem_comp_1.loops.get("Chem_comp_SMILES", None)
    if smilesTable:
        for row in smilesTable.rows:
            if row[0] == "canonical":
                entry.smiles = row[1]
                break
    
    entry.inchi = chem_comp_1.datums.get("InChI_code", None)
    if entry.inchi is not None:
        inchi = entry.inchi.split("=")[1]
        
    entry.CASRegistries = set()
    entry.PubChemCodes = set()
    chem_comp_1_name = chem_comp_1.datums["Name"]
    entry.molNames = {chem_comp_1_name}
    
    linkTable = chem_comp_1.loops.get("Chem_comp_db_link", None)
    if linkTable is None:
        print("  No db link table found.")
    else:
        CASKeys = ["CAS Registry", "CAS"]
        PubChemKeys = ["PubChem"]
        df = loopToDataFrame(linkTable)
        entry.PubChemCodes = list(df.loc[np.isin(df["Database_code"], PubChemKeys), "Accession_code"])
        entry.CASRegistries = list(df.loc[np.isin(df["Database_code"], CASKeys), "Accession_code"])
        entry.molNames = set(df["Entry_mol_name"]) if "Entry_mol_name" in df.columns else set()
            
    print("   Nr of molecule names: %d"%len(entry.molNames))
    print("     names:", entry.molNames)
    if "." in entry.molNames:
        entry.molNames = entry.molNames.difference(".")
        print("      Removing '.' from names...")
    print("   Nr of CAS registries: %d"%len(entry.CASRegistries))
    print("     codes:", entry.CASRegistries)
    print("   Nr of PubChem codes: %d"%len(entry.PubChemCodes))
    print("     codes:", entry.PubChemCodes)
    print("InChI code: %s"%entry.inchi)
    print("SMILES code: %s"%entry.smiles)
    

def parseBMRBEntry(fn):    
    with open(fn, "r") as f:
        nmrStarStr = f.read()
    p = nmrpystar.parse(nmrStarStr)       
    data = p.value.saves
    
    entry = Namespace(peaks={})
    addBasicEntryInformation(p, entry)

    if entry.name is None:
        entry.msg = "Couldn't pars substance for file '%s'"%fn
        print(entry.msg)
        return False, entry
    
    print("parseBMRBEntry()\n  Parsed NMRstar data for compound '%s' ('%s') from '%s'"%(entry.title, entry.name, fn))
    print(f"  NMR-STAR version: {entry.version} (orig: {entry.orig_version})")

    # Check if experiment_list registers a HSQC experiment
    HSQC_exp_available = reportExperimentList(data)
    
    if DEBUG:
        if entry.version not in known_versions:
            print(f"new version '{entry.version}'")
            sys.exit()
            
    
    if "chem_comp_1" not in data:
        entry.msg = f"  Couldn't find compound information for '%s' (NMR-START version {entry.version}) - skipping entry."%entry.name
        print(entry.msg)
        return False, entry
    
    addCompoundInformation(data, entry)
    
    specKey = findSpecKey(data)
    if specKey is None:
        # Try to find an alternative key for the HSQC spectrum
        print(f"  None of the specified HSQCKeys present in data. NMR-STAR version {entry.version}")
        specKey = guessUnknownSpecKey(data)
    
    entry.specKey = specKey    
    spec = None if specKey is None else data[specKey]
    
    if spec is None:
        print(f"  No peak (HSQC) data found for '%s' (NMR-STAR version '{entry.version}')"%entry.name)
        print(f"     (HSQC experiment mentioned in experiment_list? {HSQC_exp_available})")
        # Debug
        if DEBUG:
            if HSQC_exp_available:
                print("HSQC experiment mentioned but not found.")
                sys.exit()
    else:
        shiftTable = None
        for k in shiftTableKeys:
            shiftTable = spec.loops.get(k, None)
            if shiftTable is not None:
                print("  Using shift table key '%s'"%k)
                shiftTableKey = k
                entry.shiftTableKey = k
                break
        if shiftTable is None:
            entry.msg = "Found no shift table for '%s'"%entry.name
            print(entry.msg)
            return True, entry
        for i, _ in enumerate(shiftTable.rows):
            d = shiftTable.getRowAsDict(i)
            pid = d[peakIDField[shiftTableKey]]
            entry.peaks.setdefault(pid, {})
            dim = d["Spectral_dim_ID"]
            entry.peaks[pid].setdefault(dim, [])
            entry.peaks[pid][dim].append(d[chemShiftField[shiftTableKey]])
        print("   Nr of peaks: %d"%len(entry.peaks))
        for i, pid in enumerate(sorted(entry.peaks.keys())):
            print("     peak %d: %s"%(i, entry.peaks[pid]))
    return True, entry


def parseBMRBEntries(parallel=True):
    file_list = list(DOWNLOAD_DIR.iterdir())
    offset = 0
    if parallel:
        pool = mp.Pool()
        results = pool.map(parseBMRBEntry, file_list[offset:])
    else:
        results = map(parseBMRBEntry, file_list[offset:])
    entries = []
    failed = []
    for i, res in enumerate(results):
        print("\nEntry ", i+offset)
        success, entry = res
        if not success:
            print(f"Parsing failed for '{file_list[offset+i]}'.")
            failed.append(file_list[offset+i])
            continue
        pp(entryAsDict(entry))
        entries.append(entry)
    return entries, failed
        
    
def entryAsDict(e):
    attrs = [attr for attr in dir(e) if attr[0]!="_"]
    return {a:getattr(e, a) for a in attrs}


def spectrumFromEntry(e):
    print(f"\nEntry {e.name}")
    pp(e.peaks)
    df = pd.concat([pd.DataFrame(p) for p in e.peaks.values()])
    
    if not np.all(np.isin(["1", "2"], list(df.columns))):
        print(f"peak ids '1' or '2' not found for {e.name}")
        if ("?" in list(df.columns)):
            print(f"Spectrum for {e.name} might be incomplete... skipping.")
            return None
        print(df)
        if len(df.columns) < 2:
            return None
        else:
            sys.exit(1)
    
    coordsH, coordsC = [float(c) for c in df["1"]], [float(c) for c in df["2"]]
    
    # Debug
    if np.mean(coordsH) > np.mean(coordsC):
        print(f"coord order seems to be wrong for {e.name}. Switching '1' for '2'.")
        print(df)
        coordsH, coordsC = coordsC, coordsH
        
    coords = np.vstack((coordsC, coordsH)).T
    ptspec = PointSpectrum(coords, weights=np.ones_like(coordsH), name=e.name)
    return ptspec
    

def makeBMRBLib(entries):
    lib = {}
    nopeaks = []
    for e in entries:
        if e.peaks:
            if e.name in lib:
                print(f"Duplicate entry '{e.name}'!")
                print("Version 1:")
                print(lib[e])
                print("Version 2:")
                pp(entryAsDict(e))
                # Debug
                if DEBUG:
                    sys.exit(1)
            spec = spectrumFromEntry(e)
            if spec is None:
                nopeaks.append(e.name)
            else:
                lib[e.name] = spec
        else:
            nopeaks.append(e.name)
    return lib, nopeaks


def prepareBMRB():
    a = 0
    while a==0:
        a = input("\nPrepare BMRB lib? (This takes a few minutes) (Y/n) ")
        if a.strip() in ["", "Y", "y", "yes"]:
            break
        elif a.strip() in ["n", "N", "no"]:
            return
        else:
            a=0

    libfile = BMRB_DIR / "BMRBlib.csv"     
    if libfile.exists():
        a=0
        while a==0:
            a = input(f"File '{libfile}' already exists.\nRe-generate? (Y/n) ")
            if a.strip() in ["", "Y", "y", "yes"]:
                break
            elif a.strip() in ["n", "N", "no"]:
                return
            else:
                a=0
    
    downloadBMRB()
    entries, failed = parseBMRBEntries()
    
    # # Save intermediate results for faster debug
    # entries_fn = get_mcfnmr_home() / "tmp/entriesBMRB.pickle"
    # with open(entries_fn, "wb") as f:
    #     pickle.dump((entries, failed), f)
    # print(f"Saved entries to '{entries_fn}'")
    
    # with open(entries_fn, "rb") as f:
    #     entries, failed = pickle.load(f)
        
    print(f"\nParsed {len(entries)} entries from {len(entries)+len(failed)} files (failed: {len(failed)})")    
    
    lib, nopeaks = makeBMRBLib(entries)
    
    print(f"\nCreated library of {len(lib)} compounds, didn't find peaks for {len(nopeaks)} entries.")
    
    saveLibAsCSV(lib, "BMRBlib", libfile)
    
    print("\nPreparing BMRB done.\n")

if __name__=="__main__":
    prepareBMRB()