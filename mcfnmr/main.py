import sys
import os
import argparse
import toml
from pathlib import Path
import gzip
import pandas as pd
import numpy as np
import matplotlib as mpl

from copy import deepcopy
from mcfnmr.utils.pointspectrum import PointSpectrum
from mcfnmr.core.mcf import mcf
from mcfnmr import __version__
from mcfnmr.config import REPO_URL
from mcfnmr.routines.utils import (
    singleRun,
    incrementalMCFResultsUpdate,
    updateSavedMCFResult,
)
from mcfnmr.utils.plotting import plot_detected
from pprint import pp


def parse_args():
    arg_parser = argparse.ArgumentParser(
        prog="mcfNMR",
        description=f"""
        MCFNMR  (v{__version__}). A Minimum Cost Flow NMR recombinator.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "--config",
        "-c",
        dest="config",
        help=f"Configuration file for the run. See README.",
        required=True,
        type=str,
    )
    return arg_parser.parse_args()


def check_config_consistency(config):
    config_fn = config["config_path"]
    config_dir = config_fn.parent
    config.setdefault("isolated_fit", False)
    config.setdefault("incremental_fit", False)
    config.setdefault("output_dir", config_dir / "mcfNMR_output")
    config.setdefault("output_file", None)
    config.setdefault("load", False)
    config.setdefault("plot", False)
    config.setdefault("show_plot", False)
    config.setdefault("gfxformat", "png")

    # Obligatory
    for key in ["lib", "target", "assignment_radius", "detection_threshold"]:
        if key not in config:
            raise Exception(
                f"Config must contain entry '{key}'. Not found in {config_fn}"
            )

    try:
        config["detection_threshold"] = float(config["detection_threshold"])
    except:
        raise Exception(
            f"Couldn't parse float from entry 'detection_threshold' in '{config_fn}': {config['detection_threshold']}"
        )
    try:
        config["output_dir"] = Path(config["output_dir"])
        if not config["output_dir"].is_absolute():
            config["output_dir"] = config_dir / config["output_dir"]
    except:
        raise Exception(
            f"Couldn't parse Path from entry 'output_dir' in '{config_fn}': {config['output_dir']}"
        )

    # Rigid type check
    types = [
        (str, ["lib", "target", "name"]),
        (bool, ["isolated_fit", "incremental_fit", "load"]),
    ]
    for typ, entries in types:
        for e in entries:
            if type(config[e]) is not typ:
                t = type(config[e])
                raise Exception(
                    Exception(
                        f"Entry '{e}' in '{config_fn}' must be of type '{typ}'."
                        f"Found {t}: {config[e]}"
                    )
                )

    for fn in ["target", "lib"]:
        config[fn] = Path(config[fn])
        if not config[fn].is_absolute():
            config[fn] = config_dir / config[fn]

    # Check assignment radius for incremental_fit=True/False
    assignment_radius = config["assignment_radius"]
    if config["incremental_fit"]:
        # assignment_radius should be list of floats
        if type(assignment_radius) is not list:
            raise Exception(
                "For incremental fit, 'assignment_radius' must be a list of floats. "
                f"Found: {assignment_radius} in '{config_fn}'."
            )
        try:
            config["assignment_radius"] = [
                float(r) for r in config["assignment_radius"]
            ]
        except:
            raise Exception(
                "Could not parse floats for all elements of "
                f"'assignment_radius' in '{config_fn}': {assignment_radius}"
            )
    else:
        # assignment_radius should single float
        try:
            config["assignment_radius"] = float(assignment_radius)
        except:
            raise Exception(
                "Could not parse float from "
                f"'assignment_radius' in '{config_fn}': {assignment_radius}"
            )

    supported_gfxformats = mpl.backend_bases.FigureCanvasBase.get_supported_filetypes()
    if config["gfxformat"] not in supported_gfxformats:
        print(
            f"Graphics format '{config['gfxformat']}' is not supported.\nTry one of these:"
        )
        pp(supported_gfxformats)
        raise Exception("Unsupported value for 'gfxformat'.")

    # Hard-coded normalize_assignment (that is, the quantity which is finally compared
    # to the detection_threshold is the assignment to a compound relative to the total
    # weight of the target)
    config["normalize_assignment"] = True


def load_lib(fn):
    fn = Path(fn).absolute()

    if not fn.exists():
        raise Exception(f"File '{fn}' does not exist.")

    gzipped = fn.name[-3:] == ".gz"
    if gzipped:
        libname = ".".join(fn.name.split(".")[:-2])
        f = gzip.open(fn)
    else:
        libname = ".".join(fn.name.split(".")[:-1])
        f = open(fn)
    df = pd.read_csv(f)
    f.close()

    # Treat columns case-insensitive
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    required_cols = ["name", "1H", "13C"]
    for c in required_cols:
        if c.lower() not in [col for col in df.columns]:
            raise Exception(
                f"Couldn't find required column '{c}' in library-file '{fn}'."
            )
        elif c != "name":
            df.loc[:, c.lower()] = pd.to_numeric(df.loc[:, c.lower()])

    has_weights = "weight" in df.columns
    if has_weights:
        df.loc[:, "weight"] = pd.to_numeric(df.loc[:, "weight"])

    cpds = df.groupby("name").groups
    print(f"Found %d compounds in library {libname}" % (len(cpds)))

    lib = {}
    for c, ix in cpds.items():
        dfc = df.loc[ix, :]
        if has_weights:
            weights = np.array(dfc.loc[:, "weight"])
        else:
            weights = np.ones(len(dfc))
        coords = np.vstack((dfc["13c"], dfc["1h"])).T
        ptspec = PointSpectrum(coords=coords, weights=weights, name=c)
        lib[c] = ptspec

    return lib, libname


def load_target(fn):
    fn = Path(fn).absolute()

    if not fn.exists():
        raise Exception(f"File '{fn}' does not exist.")

    gzipped = fn.name[-3:] == ".gz"
    if gzipped:
        name = ".".join(fn.name.split(".")[:-2])
        f = gzip.open(fn)
    else:
        name = ".".join(fn.name.split(".")[:-1])
        f = open(fn)
    df = pd.read_csv(f)
    f.close()

    # Treat columns case-insensitive
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    required_cols = ["1H", "13C"]
    for c in required_cols:
        if c.lower() not in [col for col in df.columns]:
            raise Exception(
                f"Couldn't find required column '{c}' in target-file '{fn}'."
            )
        else:
            df.loc[:, c.lower()] = pd.to_numeric(df.loc[:, c.lower()])

    if "weight" in df.columns:
        weights = np.array(pd.to_numeric(df.loc[:, "weight"]))
    else:
        weights = np.ones(len(df))
    coords = np.vstack((df["13c"], df["1h"])).T
    ptspec = PointSpectrum(coords=coords, weights=weights, name=name)
    return ptspec, name


def make_point_spectrum(entry, name, ensure_positivity=True):
    if ensure_positivity:
        weights = [max(0.0, float(w)) for w in entry["weights"]]
    else:
        weights = [float(w) for w in entry["weights"]]
    coords = [[float(v) for v in c] for c in entry["coords"]]
    p = PointSpectrum(coords, weights, name=name)
    return p


def make_output_filename(config, outdir, target_name, libname):
    if "name" in config:
        savefn = outdir / ("mcfresult_" + config["name"] + ".pickle")
    else:
        incrstr = "incr" if config["incremental_fit"] else "onepass"
        isostr = "iso" if config["isolated_fit"] else "joint"
        setupstr = f"({isostr}-{incrstr})"
        ar = config["assignment_radius"]
        if config["incremental_fit"]:
            savefn = f"mcfresult_{target_name}_by_{libname}{setupstr}_ar{ar:g}.pickle"
        else:
            # This base name will be used at each incremental step
            savefn = f"mcfresult_{target_name}_by_{libname}{setupstr}.pickle"
    return savefn


def make_mcf_parameters(config, target_name, libname):
    outdir = config["output_dir"]
    if not outdir.is_absolute():
        config_dir = config["config_path"].parent
        outdir = config_dir / outdir
    savefn = make_output_filename(config, outdir, target_name, libname)

    pars = dict(
        assignment_radius=config["assignment_radius"],
        absorption_cost=None,
        dist_pars={},
        target_id=target_name,
        lib_id=libname,
        savefn=savefn,
        target_regions=None,
        isolated_fit=config["isolated_fit"],
        resolveYinResult=False,
        load=config["load"],
        load_dists=config["load"],
    )
    return pars


def classify_result(result, th, verb=0):
    df = dict(
        compound=[],
        # npeaks=[],
        assigned=[],
        avg_cost=[],
        detection=[],
        original_weight=[],
    )
    compound_ids = sorted(result.compounds)

    for i, cid in enumerate(compound_ids):
        df["compound"].append(cid)
        # df["npeaks"].append(len(result.compounds[cid]["ix"]))
        df["assigned"].append(result.assigned[i])
        df["avg_cost"].append(result.specificCosts[i])
        df["original_weight"].append(result.compounds[cid]["total_weight"])
        df["detection"].append(result.assigned[i] >= th)
        if verb:
            print(f"\ncompound: {cid}")
            print(f"assigned: {result.assigned[i]}")
            print(f"detected: {result.assigned[i] >= th}")
            print(f"threshold: {th}")

    return pd.DataFrame(df)


def save_as_text(df, savefn):
    f = open(savefn, "w")
    df.to_csv(f)
    print(f"Saved results table to '{savefn}'")
    f.close()


def incremental_series(
    target_spectrum,
    library,
    pars,
    load,
    verb=False,
):
    pars = deepcopy(pars)

    pars["absorptionCost"] = None
    pars["specificTargetSpecID"] = target_spectrum.name
    pars["libID"] = f"lib_{len(library)}cpds"
    # Initial setup for AR steps.
    # This triggers incremental fit in singleRun()
    pars["previousARSteps"] = []
    pars["reservedSinkCapacity"] = np.zeros_like(target_spectrum.weights)
    prevResult, prevSavefn = None, None
    results = {}
    for ar in sorted(pars["assignment_radius"]):
        pars["assignmentRadius"] = ar
        prevStepsStr = (
            "[" + ",".join(["%g" % ar for ar in pars["previousARSteps"]]) + "]"
        )
        basefn = ".".join(str(pars["savefn"]).split(".")[:-1])
        savefn = Path(basefn + f"_ar{ar:g}on{prevStepsStr}.pickle")
        result = singleRun(
            pars, library, target_spectrum, load=load, verb=verb, savefn=savefn
        )
        inflowsY = getattr(result, "inflowsY", None)
        loadedResult = inflowsY is None
        if not loadedResult:
            # Update reserved capacity, if not loading results
            incrementalMCFResultsUpdate(result, inflowsY, prevResult)
            # Delete inflowsY and residualsY from result to save storage.
            # We will only need the reserved capacity at the next
            # larger ar-value to extend an incremental series
            result.inflowsY = None
            result.residualsY = None
            updateSavedMCFResult(result, savefn=savefn)
            if prevResult:
                # reservedCapacity not needed for non-maximal ar
                prevResult.reservedSinkCapacity = None
                updateSavedMCFResult(prevResult, savefn=prevSavefn)
        prevSavefn = savefn
        results[ar] = result
        pars["previousARSteps"].append(ar)
        pars["reservedSinkCapacity"] = getattr(result, "reservedSinkCapacity", None)
        prevResult = result
    return results


def load_config(args):
    print(f"\nMCFNMR, version {__version__}")
    config_fn = Path(args.config).absolute()
    config = toml.load(config_fn)
    if getattr(args, "outdir", None) is not None:
        # Only used by test_runner.py, not exposed
        config["output_dir"] = Path(args.outdir).absolute()
        print("\nOption '--outdir' given. Overrides config entry 'output_dir'.")
    config["config_path"] = config_fn
    return config


def run(config):
    mcf_setup = load(config)

    mcf_result = run_mcf(target_spec=mcf_setup["target_spec"], 
            lib=mcf_setup["lib"],
            mcf_pars=mcf_setup["pars"],
            incremental_fit=config["incremental_fit"],
            load=config["load"])
    detection_result = run_detection(mcf_result, 
                                     th=config["detection_threshold"], 
                                     normalize_assignment=config["normalize_assignment"])
    save_as_text(detection_result, config["output_file"])

    if config["plot"]:
        outdir = config["output_file"].parent
        figname = (
            ".".join(config["output_file"].name.split(".")[:-1])
            + "_detections."
            + config["gfxformat"]
        )
        assignment_radius = config["assignment_radius"]
        if config["incremental_fit"]:
            assignment_radius = assignment_radius[-1]
        plot_detected(
            detection_result,
            mcf_setup["lib"],
            mcf_setup["target_spec"],
            assignment_radius,
            outdir / figname,
            mcf_setup["libname"],
            show=config["show"],
        )

def load(config):
    check_config_consistency(config)

    target_spec, target_name = load_target(config["target"])
    nr_neg = np.count_nonzero(target_spec.weights < 0)
    if nr_neg:
        print(f"Setting {nr_neg} negative peak weights to zero.")
        target_spec.weights = np.maximum(target_spec.weights, 0.0)
    lib, libname = load_lib(config["lib"])

    pars = make_mcf_parameters(config, target_name, libname)

    if config["output_file"] is None:
        config["output_file"] = ".".join(pars["savefn"].name.split(".")[:-1] + ["csv"])
        if config["output_dir"] is None:
            config["output_dir"] = pars["savefn"].parent
        config["output_file"] = config["output_dir"] / config["output_file"]

    if not pars["savefn"].parent.exists():
        os.makedirs(pars["savefn"].parent)
        print(f"Created directory '{pars['savefn'].parent}'")
    if not config["output_file"].parent.exists():
        os.makedirs(config["output_file"].parent)
        print(f"Created directory '{config['output_file'].parent}'")
    
    return dict(pars=pars, target_spec=target_spec, lib=lib, libname=libname)


def run_mcf(target_spec, lib, mcf_pars, incremental_fit, load):
    if incremental_fit:
        results = incremental_series(
            target_spectrum=target_spec,
            library=lib,
            pars=mcf_pars,
            load=load,
        )
        result = results[max(results.keys())]
    else:
        result = mcf(
            target_spectrum=target_spec,
            library=lib,
            **mcf_pars,
        )
    return result


def run_detection(result, th, normalize_assignment):
    if normalize_assignment:
        th *= result.originalWeightY
    df = classify_result(result, th)
    return df
    

def main():
    try:
        args = parse_args()
        config = load_config(args)
        run(config)
    except Exception as e:
        print(f"\nmcfNMR failed with error:\n  {e}")
        sys.exit(1)

