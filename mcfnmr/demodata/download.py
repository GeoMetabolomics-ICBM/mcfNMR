#! /usr/bin/python

import sys
from .oldbdata import prepareOLDBdata
from .metabominer import prepareMM
from .bmrb import prepareBMRB

def main():
    if len(sys.argv) == 1:
        print("Please specify, which dataset to download. ('bmrb', 'oldb', 'metabominer', 'all')")
        sys.exit(1)
    elif "all" in sys.argv:
        request = ["oldb", "bmrb", "metabominer"]
    else:
        request = sys.argv[1:]
    if "oldb" in request:
        prepareOLDBdata()
    if "metabominer" in request:
        prepareMM()
    if "bmrb" in request:
        prepareBMRB()



if __name__ == "__main__":
    main()