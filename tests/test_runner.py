#! /usr/bin/python

import os
import sys
import tomllib
import argparse
import time
from pathlib import Path
from pprint import pprint
from argparse import Namespace

import mcfnmr
from mcfnmr.utils import spec2csv
from mcfnmr.utils.system import get_mcfnmr_home
from mcfnmr.routines import (
    scan,
    comparison_classification,
    comparison_quantification,
    classification,
    quantification,
    analysis_plasma_samples, test1D,
)

sys.path.append(os.environ["MCFNMR_HOME"])
os.environ["PYTHONPATH"] = (
    os.environ.get("PYTHONPATH", "") + ":" + os.environ["MCFNMR_HOME"]
)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        prog="mcfNMR test runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "--outdir",
        "-o",
        default=None,
        dest="outdir",
        help="Output base directory for generated files.",
        required=True,
        type=str,
    )
    arg_parser.add_argument(
        "--config",
        "-c",
        default=None,
        dest="config",
        help="Configuration and parameter file for the run",
        required=True,
        type=str,
    )
    return arg_parser.parse_args()


def run(args):
    MCFNMR_HOME = get_mcfnmr_home()

    # Run directory
    rundir = Path(os.getcwd())
    print(f"\nWorking directory: {rundir}")

    # Load config
    fn = Path(args.config)
    with open(fn, "rb") as file:
        cfg = tomllib.load(file)
    print("\nconfig:")
    pprint(cfg)

    # Output directory
    outdir = Path(args.outdir)
    if outdir != outdir.absolute():
        outdir = rundir / outdir
    if not outdir.exists():
        os.makedirs(outdir)
    print(f"\nWriting output to: {outdir}")
    cfg["outdir"] = outdir

    # Data directory
    datadir = MCFNMR_HOME / "data"
    print(f"\nData directory: {datadir}")
    if not datadir.exists():
        raise Exception(f"Couldn't find data directory '{datadir}'")
    cfg["datadir"] = datadir

    # Debug
    # Whether to recompute all parts of the result
    # or use cache for the run (speeds up a lot, but
    # should be turned off for full testing)
    # cfg["recompute"] = True
    # cfg["recompute"] = False

    # Run task
    start_time = time.time()
    if cfg["task"] == "classification":
        classification.main(cfg)
    elif cfg["task"] == "quantification":
        quantification.main(cfg)
    elif cfg["task"] == "scan":
        scan.main(cfg)
    elif cfg["task"] == "test1D":
        test1D.main()
    elif cfg["task"] == "comparison_classification":
        comparison_classification.main(cfg)
    elif cfg["task"] == "comparison_quantification":
        comparison_quantification.main(cfg)
    elif cfg["task"] == "plasma_sample_analysis":
        analysis_plasma_samples.main(
            load=not cfg.get("recompute", False), 
            plot=cfg.get("plot", False),
            libName = cfg.get("lib", "MetaboMiner-all"),
        )
    elif cfg["task"] == "spec2csv":
        if cfg.get("help", False):
            sys.argv = ["spec2csv", "--help"]
            spec2csv.main()
        else:
            args = Namespace(
                infile=str(MCFNMR_HOME / cfg["target"]),
                outfile=str(outdir / cfg["outfile"]),
                bindims=cfg["bindims"],
                xrange=None,
                yrange=None,
                irange=None,
                override=False,
                sqrt=False,
                peaklist=cfg.get("peaklist", False),
            )
            args, msg = spec2csv.process_args(args)
            if args is None:
                raise Exception(f"Problem with test config: '{msg}'")
            spec2csv.run(args)
    elif cfg["task"] == "mcfNMR":
        if cfg.get("help", False):
            sys.argv = ["mcfNMR", "--help"]
            mcfnmr.main()
        else:
            args = Namespace(
                config=get_mcfnmr_home() / cfg["config"],
                outdir=outdir,
            )
            config = mcfnmr.load_config(args)
            for k, v in cfg.items():
                if k in ["outdir", "config"]:
                    continue
                config[k] = v
            mcfnmr.run(config)
    else:
        raise Exception(f"Unknown task '{cfg['task']}'")
    elapsed = time.time() - start_time
    print(f"\n Total run time: {elapsed} secs")


def main(config_fn=None):
    if config_fn is None:
        print("Test run ...")
        args = parse_args()
        print("\nargs:")
        print(args)
    else:
        args = Namespace(
            config=config_fn,
            outdir=get_mcfnmr_home() / "output" / "tmp",
        )
    run(args)


if __name__ == "__main__":
    # Debug example
    # main(get_mcfnmr_home() / "tests" / "user_interface" / "mcfNMR" / "mcfNMR_N988_grid_incremental" / "test_config.toml")
    main()
