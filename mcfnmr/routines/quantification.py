import os
import pickle
import numpy as np
from copy import deepcopy
from pprint import pprint

from mcfnmr.config import OUTDIR
from mcfnmr.demodata import case_dict
from mcfnmr.demodata.loading import loadOLDBCompoundLib, loadOLDBMixSpectrum
from mcfnmr.utils.pointspectrum import makeRasterPointSpectrum
from mcfnmr.demodata import OLDBCompoundWeights
from mcfnmr.routines.utils import (
    cache_dir,
    singleRun,
    incrementalMCFResultsUpdate,
    prepareParameters,
    makeResultsDF,
    makeMCFResultFilename,
    updateSavedMCFResult,
)
from mcfnmr.routines.plotting import plotAssignmentStatsOLDB

quantification_cache_dir = cache_dir / "quantification"
if not quantification_cache_dir.exists():
    os.makedirs(quantification_cache_dir)
    print(f"Created directory '{quantification_cache_dir}'")

# Random selection of the 34 compounds to show in barplots
showCompounds = sorted(
    [
        "CinA",
        "Bnz3",
        "HCin",
        "PheB",
        "Bio",
        "Gluc",
        "Man",
        "Met",
        "Pro",
        "Rha",
        "Tyr",
        "Nic",
        "Pim",
        "Van",
    ]
)


def makeStatsFilename(
    targetSpecID,
    binning,
    isolated_fit,
    incrementalFit,
    pointTarget,
    libID,
    assignmentRadiusRange,
    noiseFilterLevel,
    smoothing,
):
    saveStatsDir = os.path.join(OUTDIR, "cache")
    os.makedirs(saveStatsDir, exist_ok=True)
    fn = (
        makeSetupStr(
            targetSpecID,
            binning,
            isolated_fit,
            incrementalFit,
            pointTarget,
            libID,
            assignmentRadiusRange,
            noiseFilterLevel,
            smoothing,
        )
        + ".pickle"
    )
    return os.path.join(saveStatsDir, fn)


def makeSetupStr(
    targetSpecID,
    binning,
    isolated_fit,
    incrementalFit,
    pointTarget,
    libID,
    assignmentRadiusRange,
    noiseFilterLevel,
    smoothing,
):

    isoStr = "" if isolated_fit is None else ("iso" if isolated_fit else "joint")
    incrStr = "incrFit" if incrementalFit else ""
    pointTargetStr = "" if pointTarget is None else ("pt" if pointTarget else "rst")
    noiseFilterStr = (
        "" if noiseFilterLevel is None else "noiseFilter%g" % noiseFilterLevel
    )
    smoothingStr = "_smooth%g" % smoothing
    arRangeStr = "[" + ", ".join(["%g" % ar for ar in assignmentRadiusRange]) + "]"
    return f"{targetSpecID}({binning},{pointTargetStr}){incrStr}{noiseFilterStr}{smoothingStr}_lib{libID}({isoStr})_ar{arRangeStr}"


def saveStats(fn, stats):
    with open(fn, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved  '{fn}'")


def loadStats(fn):
    with open(fn, "rb") as f:
        stats = pickle.load(f)
    print(f"Loaded '{fn}'")
    return stats


def single_quantification(pars, lib, targetPointSpec, load=True, verb=1):
    savefn = quantification_cache_dir / makeMCFResultFilename(
        pars, pars["targetSpecID"]
    )
    res = singleRun(
        pars=pars, lib=lib, targetPointSpec=targetPointSpec, savefn=savefn, load=load
    )
    return res


def runSingleQuantificationIncremental(pars, arSpan, lib, load=True, return_all=False):
    targetSpec = loadOLDBMixSpectrum(
        specID=pars["targetSpecID"],
        smooth=pars["smoothing"],
        noiseFilterLevel=pars["noiseFilterLevel"],
    )
    targetPointSpec = makeRasterPointSpectrum(
        rasterspec=targetSpec, nbin=pars["binning"]
    )
    targetPointSpec.weights = np.maximum(targetPointSpec.weights, 0.0)

    results = []
    prevResult = None
    prevSavefn = None
    # This reserves assigned sink capacity after each run
    pars["previousARSteps"] = []
    pars["reservedSinkCapacity"] = np.zeros_like(targetPointSpec.weights)
    for ar in arSpan:
        pars = deepcopy(pars)
        pars["assignmentRadius"] = ar
        savefn = quantification_cache_dir / makeMCFResultFilename(
            pars, pars["targetSpecID"]
        )
        res = singleRun(
            pars=pars,
            lib=lib,
            targetPointSpec=deepcopy(targetPointSpec),
            savefn=savefn,
            load=load,
        )
        loaded_res = res.inflowsY is None
        # Update sequence memory
        pars["previousARSteps"].append(ar)
        if (
            not loaded_res
            and np.abs(np.round(sum(res.inflowsY) - sum(res.assigned), 6)) > 0.0
        ):
            raise Exception(
                "runSingleQuantificationIncremental() total target inflows must equal sum of individual assignments!"
            )
        if not loaded_res:
            # Add up assigned flow from previous runs
            incrementalMCFResultsUpdate(res, res.inflowsY, prevResult)
        pars["reservedSinkCapacity"] = res.reservedSinkCapacity
        # Delete inflowsY and residualsY from result to save storage.
        # We will only need the reserved capacity at the next
        # larger ar-value to extend an incremental series
        res.inflowsY = None
        res.residualsY = None
        updateSavedMCFResult(res, savefn=savefn)
        if prevResult:
            # reservedCapacity not needed for non-maximal ar
            prevResult.reservedSinkCapacity = None
            updateSavedMCFResult(prevResult, savefn=prevSavefn)
        results.append({"res": res, "pars": pars})
        prevSavefn = savefn
        prevResult = res
    if return_all:
        return results
    else:
        return results[-1]


def runSingleQuantification(pars, ar, lib, load=True):
    targetSpec = loadOLDBMixSpectrum(
        specID=pars["targetSpecID"],
        smooth=pars["smoothing"],
        noiseFilterLevel=pars["noiseFilterLevel"],
    )
    targetPointSpec = makeRasterPointSpectrum(
        rasterspec=targetSpec, nbin=pars["binning"]
    )
    pars = deepcopy(pars)
    pars["assignmentRadius"] = ar
    savefn = quantification_cache_dir / makeMCFResultFilename(
        pars, pars["targetSpecID"]
    )
    res = singleRun(
        pars=pars,
        lib=lib,
        targetPointSpec=deepcopy(targetPointSpec),
        savefn=savefn,
        load=load,
    )
    result = {"res": res, "pars": pars}
    return result


def main(cfg):
    assert cfg["task"] == "quantification"
    assert cfg["lib"].lower() == "inhouse"
    mix = cfg["mix"].split(".")
    assert len(mix) == 2
    series, experiment = mix
    assert series in ["I", "II", "III"]
    assert experiment in ["a", "b", "c"]
    assert cfg["setup"] in ["A", "B", "C"]

    # For incremental assignment
    arSpan = np.array(
        [
            0.025,
            0.05,
            0.075,
            0.1,
            0.125,
            0.15,
            0.175,
            0.2,
            0.225,
            0.25,
            0.3,
            0.35,
            0.4,
            0.5,
        ]
    )

    # OLDB compound lib
    lib = loadOLDBCompoundLib()
    setupSpec = case_dict[cfg["setup"]][cfg["mix"]]
    isolated_fit = setupSpec["isolated_fit"]
    incrementalFit = setupSpec["incrementalFit"]
    pointTarget = setupSpec["pointTarget"]

    # All individual compounds with concentrations 30mM
    libID = "OLDBcompoundLib"
    targetSpecID = series + experiment + "_01"
    binning = (2, 4)
    noiseFilterLevel = None
    smoothing = 0
    pars = prepareParameters(
        isolated_fit=isolated_fit,
        targetSpecID=targetSpecID,
        pointTarget=pointTarget,
        libID=libID,
        binning=binning,
        specDomain="inhouse",
        incrementalFit=incrementalFit,
        noiseFilterLevel=None,
        smoothing=0,
        workingDir=cfg["outdir"],
        recompute=cfg.get("recompute", True),
    )

    load = not pars["recompute"]
    # Run
    if incrementalFit:
        ars = np.append(arSpan[arSpan < cfg["ar"]], cfg["ar"])
        res = runSingleQuantificationIncremental(pars, arSpan=ars, lib=lib, load=load)
    else:
        res = runSingleQuantification(pars, ar=cfg["ar"], lib=lib, load=load)
        ars = [cfg["ar"]]

    # This assumes that all weight of X is in the target regions
    dfTargets, dfCompounds = makeResultsDF(
        [res], intensityCorrection=True, originalWeights=OLDBCompoundWeights
    )

    setupStr = makeSetupStr(
        targetSpecID,
        binning,
        isolated_fit,
        incrementalFit,
        pointTarget,
        libID,
        ars,
        noiseFilterLevel,
        smoothing,
    )
    figs_dir = OUTDIR / "figs"
    stats = plotAssignmentStatsOLDB(
        dfCompounds,
        dfTargets,
        libID,
        targetSpecID,
        showCompounds,
        outdir=figs_dir,
        show=False,
        setupStr=setupStr,
    )
    print(f'error_stats["{targetSpecID}"] = \\')
    pprint(stats["errors"])
    fn = makeStatsFilename(
        targetSpecID,
        binning,
        isolated_fit,
        incrementalFit,
        pointTarget,
        libID,
        ars,
        noiseFilterLevel,
        smoothing,
    )
    saveStats(fn, stats)


if __name__ == "__main__":
    # Test run
    cfg = dict(
        lib="InHouse",
        mix="I.a",
        recompute=False,
        task="quantification",
        setup="C",
        ar=0.15,
        outdir=cache_dir,
    )
    main(cfg)
