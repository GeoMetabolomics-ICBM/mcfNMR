import os
import wget

from mcfnmr.config import OLDBFULLSPECTRADIR, OLDB1DTESTDIR, OLDBPEAKSDIR

def prepareOLDBdata():
    for DIR in [OLDB1DTESTDIR, OLDBFULLSPECTRADIR, OLDBPEAKSDIR]:
        if not (DIR).exists():
            os.makedirs(DIR)
            print(f"Created directory '{DIR}'")
            
    baseurl = "https://zenodo.org/records/14888536/files/"
    for s1 in ["I", "II", "III"]:
        for s2 in ["a", "b", "c"]:
            fn = f"nmr_{s1}{s2}_01.txt.gz"
            url = baseurl + fn
            target_fn = OLDBFULLSPECTRADIR / fn
            if not target_fn.exists():
                print(f"Downloading {fn} ...")
                wget.download(url, str(target_fn))
            else:
                print(f"'{target_fn}' already exists. Skipped downloading.")
            print("")
                
    # 1D example
    fn = "nmr_IIa_1H.csv.gz"
    url = baseurl + fn
    target_fn = OLDB1DTESTDIR / fn
    if not target_fn.exists():
        print(f"Downloading {fn} ...")
        wget.download(url, str(target_fn))
    else:
        print(f"'{target_fn}' already exists. Skipped downloading.")
    print("")

    # Compound spectrum peak lists
    fn = "peaklists_compounds.csv"
    url = baseurl + fn
    target_fn = OLDBPEAKSDIR / fn
    if not target_fn.exists():
        print(f"Downloading {fn} ...")
        wget.download(url, str(target_fn))
    else:
        print(f"'{target_fn}' already exists. Skipped downloading.")
    print("")

    print("\nPreparing OLDB data done.\n")


if __name__ == "__main__":
    prepareOLDBdata()
