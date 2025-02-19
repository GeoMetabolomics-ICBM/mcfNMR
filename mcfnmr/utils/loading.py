import os
import gzip
import numpy as np
import struct
import subprocess as sp

from mcfnmr.config import DEBUG, UCSFDATA_BIN


def parseUCSF(fn, outdir, cleanup=True):
    # Generate and process header

    headerFile = outdir / (fn.name + "_header.txt")
    header = getUCSFHeader(headerFile, fn)

    # Generate and process matrix
    binaryFile = outdir / (fn.name + "_matrix.m")
    matrix = getUCSFMatrix(binaryFile, fn, header)

    if cleanup:
        os.remove(headerFile)
        os.remove(binaryFile)

    return matrix, header


def parseUCSFHeader(fn):
    with open(fn, "r") as f:
        lines = f.readlines()
    header = {}
    for l in lines[2:]:
        ll = l.strip().split()
        k = ll[0].strip()
        for s in ll[1:-2]:
            k += " " + s.strip()
        header[k] = (float(ll[-2]), float(ll[-1]))
    if DEBUG > 1:
        print(header)
    # Add frequency ranges to header
    C_range = (header["upfield ppm"][0], header["downfield ppm"][0])
    H_range = (header["upfield ppm"][1], header["downfield ppm"][1])
    header["FRanges"] = (C_range, H_range)
    return header


def getUCSFHeader(headerFile, ucsfFile):
    with open(headerFile, "w") as f:
        sp.call([UCSFDATA_BIN, ucsfFile], stdout=f)
    header = parseUCSFHeader(headerFile)
    return header


def getUCSFMatrix(dataFile, ucsfFile, header):
    # "If the -m flag is present the data matrix is written in binary format as IEEE floats with your machine's native byte order.
    #  The matrix is written out with highest axis varying fastest."
    with open(dataFile, "w") as f:
        sp.call([UCSFDATA_BIN, "-m", ucsfFile], stdout=f)

    file = open(dataFile, "rb")
    nbytes = os.path.getsize(dataFile)
    n = int(nbytes / 4)
    assert n == nbytes / 4
    f = struct.unpack("f" * n, file.read(4 * n))
    f = list(reversed(f))

    mDim = header["matrix size"]
    mDim = (int(mDim[0]), int(mDim[1]))
    if DEBUG > 1:
        print("len(f)=", len(f))
        print("matrix size=", mDim[0] * mDim[1])

    assert mDim[0] * mDim[1] == len(f)
    f = np.array(f, dtype=np.float32)
    f.shape = mDim

    return f


def parseBrukerSpectrum(fn):
    # Specific lines that indicate the end of the header for the used txt formats
    headerStopSymbols = ["= Re Matrix", "= Processed Matrix", "= RR Matrix"]

    def headerEnd(line):
        for s in headerStopSymbols:
            if line[: len(s)] == s:
                return True
        return False

    headerLines = []
    with gzip.open(fn, "rb") as f:
        # Load header
        headerOpen = True
        while headerOpen:
            l = f.readline()[:-1].decode()
            headerLines.append(l)
            if headerEnd(l):
                headerOpen = False
                continue
            print(l[:-1])
        header = parseBrukerHeader(headerLines)
        print("Loaded header data for '%s'" % fn.name)

        # Read matrix data
        pointDimensions = header["pointDimensions"]
        dataStr = []
        for i in range(pointDimensions[0]):
            # print("%d/%d"%(i,self.pointDimensions[0]))
            l = f.readline().decode()
            dataStr.append(l)
    fullData = np.loadtxt(dataStr, delimiter=" ")
    print("Loaded spectrum data for '%s'" % fn.name)
    assert fullData.shape == pointDimensions

    return fullData, header


def parseBrukerHeader(lines):
    # float variable are 2D-header pairs, holding numeric information
    floatVariables = {"SweepWidth", "Frequency", "Offset", "PointsCount"}
    intVariables = {"ExpCount"}
    headerDict = dict()
    for l in lines:
        if l.count("=") == 1:
            # key value pair, probably
            k, v = l.split("=")
            k, v = k.strip(), v.strip()
            if k in floatVariables:
                v = v.split(" ")
                v = float(v[0]), float(v[1])
            if k in intVariables:
                v = int(v)
            headerDict[k] = v

    # Calculate spectral dimansions from header information and include into header
    headerDict["scanNr"] = headerDict.get("ExpCount", None)
    headerDict["pointDimensions"] = tuple(
        [int(n) for n in reversed(headerDict["PointsCount"])]
    )
    range1 = np.array(
        (
            headerDict["Offset"][0] - 0.5 * headerDict["SweepWidth"][0],
            headerDict["Offset"][0] + 0.5 * headerDict["SweepWidth"][0],
        )
    )
    range2 = np.array(
        (
            headerDict["Offset"][1] - 0.5 * headerDict["SweepWidth"][1],
            headerDict["Offset"][1] + 0.5 * headerDict["SweepWidth"][1],
        )
    )
    range1 /= headerDict["Frequency"][0]
    range2 /= headerDict["Frequency"][1]
    headerDict["FRanges"] = (range2, range1)

    return headerDict
