import argparse
import os
from pathlib import Path
import sys
from enum import Enum
import numpy as np
import pandas as pd
import gzip
from PIL import Image

from mcfnmr import __version__
from mcfnmr.utils.loading import parseBrukerSpectrum, parseUCSF
from mcfnmr.utils.rasterspectrum import RasterSpectrum
from mcfnmr.utils.pointspectrum import PointSpectrum
from mcfnmr.utils.plotting import plotRasterSpectrum
from mcfnmr.config import csv_column_name_links


class InType(Enum):
    UCSF = 1
    BRUKER = 2
    BITMAP = 3
    PEAKLIST = 4


def saveLibAsCSV(lib, libname, libfile):    
    df = {"name": [], "weight": [], "1H": [], "13C": []}
    cpds = sorted(lib.keys())
    for c in cpds:
        spec = lib[c]
        df["name"].extend([c] * spec.size())
        df["weight"].extend(spec.weights)
        df["1H"].extend(spec.coords[:, 1])
        df["13C"].extend(spec.coords[:, 0])
    pd.DataFrame(df).to_csv(libfile, sep=",")
    print(f"\nWrote lib '{libname}' to '{libfile}'.\n")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        prog="spec2csv",
        description=f"""
        spec2csv - part of MCFNMR (v{__version__}).
        Tool to convert a different types of spectrum files into a csv (or csv.gz) file usable as target for mcfnmr.
        Usage:  spec2csv --file=INPUT_FILE [--rangex=X1,X2 --rangey=Y1,Y2 --irange=I1,I2 --bindims=NX,NY]
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "--inputfile",
        "-f",
        default=None,
        dest="infile",
        help="Spectrum file describing the intensity distribution. Currently supported: "
        "ucsf-files, bruker-txt-files, and bitmaps encoding intensity as greyscale. "
        "For the latter, the options 'rangex|y', and 'irange' have to be specified.",
        required=True,
        type=str,
    )
    arg_parser.add_argument(
        "--outputfile",
        "-o",
        default=None,
        dest="outfile",
        help="Destination file for generated csv document. Defaults to '<inputfile's basename>.csv.gz'.",
        required=False,
        type=str,
    )
    ## Options for bitmap conversion (postponed for v0.2 - TODO)
    # arg_parser.add_argument(
    #     "--xrange",
    #     "-x",
    #     default=None,
    #     dest="xrange",
    #     help="For bitmap conversion: range of input x-axis. Use X1>X2 if img-axis is reversed (left coord > right coord).",
    #     required=False,
    #     type=str,
    # )
    # arg_parser.add_argument(
    #     "--yrange",
    #     "-y",
    #     default=None,
    #     dest="yrange",
    #     help="For bitmap conversion: range of input y-axis. Use Y1>Y2 if img-axis is reversed (bottom coord > top coord)",
    #     required=False,
    #     type=str,
    # )
    # arg_parser.add_argument(
    #     "--irange",
    #     "-i",
    #     default=None,
    #     dest="irange",
    #     help="For bitmap conversion: range of input's intensities."
    #     " The greyscale values are mapped linearly onto the given range.",
    #     required=False,
    #     type=str,
    # )
    # arg_parser.add_argument(
    #     "--sqrt",
    #     action="store_true",
    #     default=False,
    #     dest="sqrt",
    #     help="Flag to indicate whether the given bitmap greyscale is sqrt-transformed.",
    #     required=False,
    # )
    arg_parser.add_argument(
        "--bindims",
        "-b",
        default="1,1",
        dest="bindims",
        help="Integer number of grid points on x- and y-axis to be summarized in NXxNY bins. "
        "Determines the resolution of the input as =~original_resolution./bindims. "
        "Note: high resolutions may entail long computation times.",
        required=False,
        type=str,
    )
    arg_parser.add_argument(
        "--override",
        "-w",
        action="store_true",
        default=False,
        dest="override",
        help="Flag to indicate whether any existing destination file should be overridden.",
        required=False,
    )
    arg_parser.add_argument(
        "--peaklist",
        action="store_true",
        default=False,
        dest="peaklist",
        help="Flag to indicate whether infile is a peak list. "
        "Such should be a table with headers indicating dimensions "
        "(e.g., '1H' and '13C'), which can be resolved by comparing to config.csv_column_name_links. "
        "If peak weights are included, add a column 'weight'. "
        "File must be readable by pandas.read_csv() with default arguments, see "
        "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html",
        required=False,
    )
    return arg_parser.parse_args()


def get_input_type(file_ext):
    if file_ext == "ucsf":
        return InType.UCSF
    elif file_ext in ["txt", "txt.gz"]:
        return InType.BRUKER
    elif file_ext in ["csv", "csv.gz"]:
        return InType.PEAKLIST
    elif file_ext in ["png"]:
        return InType.BITMAP
    else:
        raise Exception(f"Couldn't derive file type from extension '{file_ext}'")


def process_args(args):
    processed_args = argparse.Namespace()
    wdir = Path(os.getcwd())
    True
    infile = Path(args.infile)
    if not infile.is_absolute():
        infile = wdir / infile
    if not infile.exists():
        return None, f"Input file '{infile}' does not exist."
    processed_args.infile = infile

    ext = infile.name.split(".")[-1]
    if ext == "gz":
        ext = ".".join(infile.name.split(".")[-2:])
    try:
        processed_args.type = get_input_type(ext)
    except:
        return None, f"Failed to map file extension '{ext}' to input type."

    if args.outfile:
        outfile = Path(args.outfile)
        if (outfile.name[-7:] != ".csv.gz") and (outfile.name[-4:] != ".csv"):
            return (
                None,
                f"Output file must have extension '.csv.gz' or '.csv'. Given: {outfile.name}",
            )
        if not outfile.is_absolute():
            outfile = wdir / outfile
    else:
        outfile = None

    if outfile is not None and outfile.exists() and not args.override:
        return (
            None,
            f"Output destination '{outfile}' already exists. You can force overriding by option --override.",
        )
    processed_args.outfile = outfile
    processed_args.override = args.override

    bindims = args.bindims
    if bindims is None:
        processed_args.bindims = None
    else:
        try:
            nX, nY = bindims.split(",")
            processed_args.bindims = int(nX), int(nY)
        except:
            return (
                None,
                f"Failed to process argument 'bindims', given value: {args.bindims}.",
            )

    ## Options for bitmap conversion (postponed for v0.2 - TODO)
    # bitmap_options = ["xrange", "yrange", "irange"]
    # if not processed_args.type == InType.BITMAP:
    #     for o in bitmap_options:
    #         if getattr(args, o) is not None:
    #             return (
    #                 None,
    #                 f"Option '{o}' is only accepted for the case of bitmap conversion.",
    #             )
    # else:
    #     for o in bitmap_options:
    #         val = getattr(args, o)
    #         try:
    #             x, y = val.split(",")
    #             setattr(processed_args, o, (float(x), float(y)))
    #         except:
    #             return None, f"Failed to process argument '{o}', given value: {val}."
    #
    # processed_args.sqrt = args.sqrt

    return processed_args, None


## Options for bitmap conversion (postponed for v0.2 - TODO)
# def bitmap2ptspec(args, name, smooth_radius=0.0, noise_cutoff=None):
#     with Image.open(args.infile) as img:
#         img.load()
#     # Grab first channel (shouldn't matter for grayscale)
#     # reverse 1st ax to have coord increase with index
#     data = np.flip(np.asarray(img)[:, :, 0], axis=0)
#
#     if np.max(data) > 1:
#         scale = 255
#     else:
#         scale = 1.0
#     data = args.irange[0] + (args.irange[1] - args.irange[0]) * data / scale
#     if args.sqrt:
#         data[data > 0] = data[data > 0] * data[data > 0]
#         data[data < 0] = -data[data < 0] * data[data < 0]
#
#     if args.yrange[0] > args.yrange[1]:
#         # Image has reversed y-axis
#         data = np.flip(data, axis=0)
#     if args.xrange[0] > args.xrange[1]:
#         # Image has reversed x-axis
#         data = np.flip(data, axis=1)
#     xrange = min(args.xrange), max(args.xrange)
#     yrange = min(args.yrange), max(args.yrange)
#     header = dict(FRanges=(xrange, yrange))
#
#     grid_spec = RasterSpectrum(
#         data,
#         header,
#         name=name,
#         denoising_pars=dict(smoothRadius=smooth_radius, noiseFilterLevel=noise_cutoff),
#     )
#
#     plotRasterSpectrum(
#         grid_spec,
#         title="Test " + name,
#         clim=(-np.max(data), np.max(data)),
#         cmap="gray",
#         show=True,
#     )
#
#     pt_spec = make_point_spectrum(grid_spec, args)
#     return pt_spec


def ucsf2ptspec(args, name, smooth_radius=0.0, noise_cutoff=None):
    matrix, header = parseUCSF(args.infile, args.infile.parent)
    grid_spec = RasterSpectrum(
        matrix,
        header,
        name=name,
        denoising_pars=dict(smoothRadius=smooth_radius, noiseFilterLevel=noise_cutoff),
    )
    pt_spec = make_point_spectrum(grid_spec, args)
    return pt_spec


def bruker2ptspec(args, name, smooth_radius=0.0, noise_cutoff=None):
    fullData, header = parseBrukerSpectrum(args.infile)
    grid_spec = RasterSpectrum(
        fullData,
        header,
        name=name,
        denoising_pars=dict(smoothRadius=smooth_radius, noiseFilterLevel=noise_cutoff),
    )
    pt_spec = make_point_spectrum(grid_spec, args)
    return pt_spec


def peaklist2ptspec(args, name):
    print(f"Parsing peak list from '{args.infile}'")
    df = pd.read_csv(args.infile)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    candis = {k: [e.lower() for e in v] for k, v in csv_column_name_links.items()}
    # For character inclusion test:
    # col_char[c] in df.column[i] => col[c] = i
    col_char = {
        "1h": "h",
        "13c": "c",
    }

    cols = {c: None for c in candis}
    for c in candis:
        for col in df.columns:
            if col in candis[c]:
                cols[c] = col
                break
        if cols[c] is None and c in col_char:
            for col in df.columns:
                # Check if this column was already used for any other dimension
                unique = not np.any(
                    [
                        col_char[c] in col
                        for col in df.columns
                        if col not in cols.values()
                    ]
                )
                if unique and col_char[c] in col:
                    cols[c] = col
                    break
        if cols[c] is None:
            print(f"Found no column for {c}.")
        else:
            print(f"Using column {cols[c]} for {c}.")

    data = {c: np.array(df.loc[:, cols[c]]) for c in cols if cols[c] is not None}
    if cols["weight"] is None:
        data["weight"] = np.ones(len(df))

    if "1H" not in data:
        print(f"Couldn't find 1H-column in peak list '{args.infile}'")
        sys.exit(1)
    if "13C" not in data:
        print(f"Couldn't find 13C-column in peak list '{args.infile}'")
        sys.exit(1)
    coords = np.vstack((np.astype(data["13C"], float), np.astype(data["1H"], float))).T
    ptspec = PointSpectrum(coords, weights=data["weight"], name=name)
    return ptspec


def make_point_spectrum(grid_spec, args):
    # Determine binning dimensions
    if args.bindims is None:
        # use input resolution
        nbin = (1, 1)
    else:
        nbin = args.bindims
        nbin = np.maximum(nbin, 1)
        name = grid_spec.name + f"{nbin}"
    coords, weights, pointClusterArea = grid_spec.getRasterDataAsPeaks(
        nbin=nbin, xVar="F1", signalThreshold=None, cutGridToBin=True
    )
    pt_spec = PointSpectrum(
        coords,
        weights,
        spec=grid_spec,
        fromRaster=True,
        binning=nbin,
        pointArea=pointClusterArea,
        name=name,
    )
    return pt_spec


def write_ptspec_as_csv(pt_spec, fn):
    if fn.name[-3:] == ".gz":
        f = gzip.open(fn, "wb")
    else:
        f = open(fn, "w")
    df = {
        "weight": pt_spec.weights,
        "1H": pt_spec.coords[:, 1],
        "13C": pt_spec.coords[:, 0],
    }
    pd.DataFrame(df).to_csv(f, sep=",")
    f.close()
    print(f"\nWritten data to {fn}.\n")


def run(args):
    infile_split = args.infile.name.split(".")
    if infile_split[-1] == "gz":
        name = ".".join(infile_split[:-2])
    else:
        name = ".".join(infile_split[:-1])
    if args.type == InType.BITMAP:
        ## Options for bitmap conversion (postponed for v0.2 - TODO)
        # pt_spec = bitmap2ptspec(args, name)
        raise Exception("Sorry, bitmap-to-spectrum conversion not implemented, yet.")
    elif args.type == InType.BRUKER:
        pt_spec = bruker2ptspec(args, name)
    elif args.type == InType.UCSF:
        pt_spec = ucsf2ptspec(args, name)
    elif args.type == InType.PEAKLIST:
        pt_spec = peaklist2ptspec(args, name)
    else:
        assert False

    # Determine outfile name
    if args.outfile is None:
        wdir = Path(os.getcwd())
        args.outfile = wdir / (name + ".csv.gz")
    if args.outfile.exists() and not args.override:
        print(f"Failed to write output to '{args.outfile}'.")
        print("File already exists. Force overriding with option '--override'.")
        sys.exit(1)

    # Save csv
    write_ptspec_as_csv(pt_spec, args.outfile)


def main():
    args = parse_args()
    processed_args, msg = process_args(args)
    if processed_args is None:
        print("\nspec2csv failed with message:\n" + msg)
        sys.exit(1)
    run(processed_args)


if __name__ == "__main__":
    main()
