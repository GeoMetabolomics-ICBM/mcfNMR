import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shutil
from pathlib import Path
from argparse import Namespace
import multiprocessing as mp

import wget
import waybackpack as wbp

from mcfnmr.config import (
    UCSFDIR,
    PROCESSED_UCSFDIR,
    METABOMINER_DIR,
    COLMAR_DATADIR,
    MM_PEAKLISTDIR,
)
from mcfnmr.demodata.loading import loadMMLibrary, loadMMMixPointSpectra
from mcfnmr.utils.loading import parseUCSF
from mcfnmr.utils import spec2csv
from mcfnmr.utils.spec2csv import saveLibAsCSV
from mcfnmr.demodata import binning

def waybackDownload(component, fn, url):
    if not fn.exists():
        print(f"Downloading {component} to '{fn}' ...")
        p = wbp.pack.Pack(url)
        p.timestamps = [p.timestamps[-1]]  # only download most recent
        p.download_to(METABOMINER_DIR / f"{component}_tmp")
        path = METABOMINER_DIR / (
            "/".join([f"{component}_tmp", p.timestamps[0]] + url.split("/")[2:])
        )
        shutil.move(path, fn)
        shutil.rmtree(METABOMINER_DIR / f"{component}_tmp")
        print("Done.")
    else:
        print(f"Skipping {component} download. {fn} already exists.")
        
    
def download_metabominer_data(ucsf_url, expls_url, mm_url, sparky_url):
    # Below, we use waybackpack to download MetaboMiner from archive.org, it uses urls like
    # "https://web.archive.org/web/20140502110649/http://wishart.biology.ualberta.ca/metabominer/downloads/..."

    ucsf_fn = METABOMINER_DIR / "ucsf.zip"
    expls_fn = METABOMINER_DIR / "examples.zip"
    mm_fn = METABOMINER_DIR / "metabominer.zip"
    sparky_fn = METABOMINER_DIR / "sparky-linux2.6-64bit.tar.gz"

    if not sparky_fn.exists():
        print(f"Downloading sparky to '{sparky_fn}'.")
        wget.download(sparky_url, str(sparky_fn))
        print("Done.")
        
    args = [
        ("MetaboMiner UCSF files", ucsf_fn, ucsf_url),
        ("MetaboMiner examples", expls_fn, expls_url),
        ("MetaboMiner", mm_fn, mm_url),
        ]
    pool = mp.Pool()
    _ = list(pool.starmap(waybackDownload, args))
    
    print("Unpacking sparky ...")
    shutil.unpack_archive(sparky_fn, METABOMINER_DIR)
    print("Unpacking ucsf files ...")
    shutil.unpack_archive(ucsf_fn, METABOMINER_DIR)
    if expls_fn.exists():
        print("Unpacking example files ...")
        shutil.unpack_archive(expls_fn, METABOMINER_DIR)
        print("Copying DEEP Picker peak lists into peak list directory ...")
        for fn in (COLMAR_DATADIR/"Deep Picker unreferenced for MCF").iterdir():
            if fn.name[-8:] == "hsqc.txt":
                shutil.copy2(fn, MM_PEAKLISTDIR / fn.name)
                print(f"   {fn.name}")
    if mm_fn.exists():
        print("Unpacking MetaboMiner files ...")
        shutil.unpack_archive(mm_fn, METABOMINER_DIR)


def save_MMlib_as_csv():
    name = "Biofluid ( all )"
    lib = loadMMLibrary(libName=name)
    libname = "MetaboMiner - " + name
    saveLibAsCSV(lib, libname, METABOMINER_DIR / "csv" / (libname+".csv"))


def process_ucsf_spectra(load=True, plot=False, show=False):
    if not (PROCESSED_UCSFDIR).exists():
        os.makedirs(PROCESSED_UCSFDIR)
        print(f"Created directory {PROCESSED_UCSFDIR}")

    files = {}
    for fn in os.listdir(UCSFDIR):
        if fn.split("_")[1] == "tocsy.ucsf":
            # Skip tocsy
            continue
        files[fn[:4]] = os.path.join(UCSFDIR, fn)

    for fid, ucsfFile in files.items():
        # Save .pickle files with matrix and header info
        print("Processing '%s'" % fid)
        saveFile = PROCESSED_UCSFDIR / (fid + ".pickle")
        if load and os.path.exists(saveFile):
            with open(saveFile, "rb") as file:
                data = pickle.load(file)
            print("Loaded data '%s'" % saveFile)
            header = data["header"]
            matrix = data["matrix"]
        else:
            matrix, header = parseUCSF(Path(ucsfFile), outdir=PROCESSED_UCSFDIR)
            header["mixID"] = fid

            # Pickling
            with open(saveFile, "wb") as file:
                pickle.dump({"matrix": matrix, "header": header}, file)
            print("Saved data '%s'" % saveFile)

        # Save grid-csvs for user interface testing
        ucsf_file = UCSFDIR / f"{fid}_hsqc.ucsf"
        bindims = binning[fid]
        outfn = METABOMINER_DIR / "csv" / f"{fid}({bindims[0]}x{bindims[1]}).csv.gz"
        if not (outfn.exists() and load):
            args = Namespace(infile=ucsf_file, bindims=bindims)
            pt_spec = spec2csv.ucsf2ptspec(args, name=fid)
            spec2csv.write_ptspec_as_csv(pt_spec, fn=outfn)

        if plot:
            plt.imshow(matrix, origin="lower")
            figname = os.path.join(PROCESSED_UCSFDIR, fid + "_matrix.png")
            plt.savefig(figname)

        if show:
            plt.show()
        else:
            plt.close("all")


def convert_peaklists():
    # Convert MetaboMiner peak lists into csv files with
    # columns 1H, 13C, and weights instead of w1, w2, and Data height
    ptSpectra = loadMMMixPointSpectra()
    for mixID in ptSpectra:
        df = {
            "weights": ptSpectra[mixID].weights,
            "1H": ptSpectra[mixID].coords[:, 1],
            "13C": ptSpectra[mixID].coords[:, 0],
        }
        fn = METABOMINER_DIR / "csv" / f"{mixID}(peaklist).csv"
        pd.DataFrame(df).to_csv(fn, sep=",")
        print(f"Wrote {fn}")


def prepareMM():
    ucsf_url = "http://wishart.biology.ualberta.ca/metabominer/downloads/ucsf.zip"
    expls_url = "http://wishart.biology.ualberta.ca/metabominer/downloads/examples.zip"
    mm_url = "http://wishart.biology.ualberta.ca/metabominer/downloads/metabominer.zip"
    if platform.system() == "Windows":
        sparky_url = "https://www.cgl.ucsf.edu/home/sparky/distrib-3.114/sparky-win32.zip"
        raise Exception("Sparky on Windows not supported yet")
    elif platform.system() == "Linux":
        sparky_url = "https://www.cgl.ucsf.edu/home/sparky/distrib-3.115/sparky-linux2.6-64bit.tar.gz"
    elif platform.system() == "Darwin":
        # MacOS
        sparky_url = "https://www.cgl.ucsf.edu/home/sparky/distrib-3.115/sparky-mac10.4-intel.dmg"
        raise Exception("Sparky on MacOS not supported yet")

    if not (METABOMINER_DIR).exists():
        os.makedirs(METABOMINER_DIR)
        print(f"Created directory {METABOMINER_DIR}")

    download_metabominer_data(ucsf_url, expls_url, mm_url, sparky_url)

    if not (METABOMINER_DIR / "csv").exists():
        os.makedirs(METABOMINER_DIR / "csv")
        print(f"Created directory {METABOMINER_DIR / 'csv'}")

    save_MMlib_as_csv()
    process_ucsf_spectra()
    convert_peaklists()

    print("\nPreparing MetaboMiner data done.\n")


if __name__=="__main__":
    prepareMM()
