from pprint import pprint
from collections import defaultdict
import numpy as np
import pickle
import os

from mcfnmr.routines.utils import (
    prepareParameters,
    makeResultsDF,
    cache_dir,
    figs_dir_oldb,
)
from mcfnmr.demodata.loading import loadOLDBCompoundLib
from mcfnmr.routines.quantification import (
    runSingleQuantificationIncremental,
    runSingleQuantification,
    makeSetupStr,
    showCompounds,
)
from mcfnmr.demodata import OLDBCompoundWeights
from mcfnmr.routines.plotting import (
    plotAssignmentStatsOLDB,
    plotErrorStats,
    plotPredictionDiagonals,
)
from mcfnmr.config import OUTDIR


def runRange(pars, arSpan, lib=None, load=True):
    incrementalFit = pars["incrementalFit"]
    if incrementalFit:
        results = runSingleQuantificationIncremental(
            pars=pars, arSpan=arSpan, lib=lib, load=load, return_all=True
        )
    else:
        results = []
        for ar in arSpan:
            res = runSingleQuantification(pars, ar=ar, lib=lib, load=load)
            results.append(res)
    return results


def main(cfg, show=False, loadtmp=False):
    
    tmpfn = OUTDIR / "tmp" / "comparison_quantification_all_dfCompounds.pickle"
    if not (OUTDIR / "tmp").exists():
        os.makedirs(OUTDIR / "tmp")
        print(f"Created directory '{OUTDIR / 'tmp'}'")
    
    assert cfg["task"] == "comparison_quantification"
    assert cfg["lib"].lower() == "inhouse"
    mix = cfg["mix"].split(".")
    assert len(mix) == 2
    series, experiment = mix
    assert series in ["I", "II", "III", "all"]
    assert experiment in ["a", "b", "c", "all"]
    assert cfg["setup"] == "all"  # This iterates through setups A, B, and C, below

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

    plot = cfg.get("plot", False)


    if loadtmp and tmpfn.exists():
        with open(tmpfn, "rb") as f:
            all_dfCompounds, data = pickle.load(f)
        print(f"Loaded all_dfCompounds from {tmpfn}")
    else:
        # OLDB compound lib
        lib = loadOLDBCompoundLib()
    
        # All individual compounds with concentrations 30mM
        libID = "OLDBcompoundLib"
        targetSpecID = series + experiment + "_01"
        binning = (2, 4)
        noiseFilterLevel = None
        smoothing = 0
        pars = prepareParameters(
            isolated_fit="all",
            targetSpecID=targetSpecID,
            pointTarget=False,
            libID=libID,
            binning=binning,
            specDomain="inhouse",
            incrementalFit="all",
            noiseFilterLevel=None,
            smoothing=0,
            workingDir=cfg["outdir"],
            recompute=cfg.get("recompute", True),
        )
        mixes = [
            "Ia_01",
            "Ib_01",
            "Ic_01",
            "IIa_01",
            "IIb_01",
            "IIc_01",
            "IIIa_01",
            "IIIb_01",
            "IIIc_01",
        ]
        # Collecting all into one dataframe
        data = defaultdict(list)
        all_dfCompounds = defaultdict(dict)
        for isoFit, incrFit in ((False, False), (False, True), (True, False)):
            for mixID in mixes:
                targetSpecID = mixID
                pars.update(
                    dict(
                        isolated_fit=isoFit,
                        targetSpecID=targetSpecID,
                        incrementalFit=incrFit,
                    )
                )
                results = runRange(pars, arSpan, lib, load=not cfg["recompute"])
    
                # This assumes that all weight of X is in the target regions
                dfTargets, dfCompounds = makeResultsDF(
                    results, intensityCorrection=True, originalWeights=OLDBCompoundWeights
                )
                all_dfCompounds[(isoFit, incrFit)][mixID] = dfCompounds
    
                restrict_arSpan = cfg.get("ar", None)
                if restrict_arSpan:
                    dfTargets = dfTargets.iloc[
                        np.isin(dfTargets["assignment radius"], restrict_arSpan)
                    ]
                    dfCompounds = dfCompounds.iloc[
                        np.isin(dfCompounds["assignment radius"], restrict_arSpan)
                    ]
                else:
                    restrict_arSpan = arSpan
    
                setupStr = makeSetupStr(
                    targetSpecID,
                    binning,
                    None,
                    None,
                    None,
                    libID,
                    restrict_arSpan,
                    noiseFilterLevel,
                    smoothing,
                )
                targetSpecID = pars["targetSpecID"]
                stats = plotAssignmentStatsOLDB(
                    dfCompounds,
                    dfTargets,
                    libID,
                    targetSpecID,
                    showCompounds,
                    outdir=figs_dir_oldb,
                    show=False,
                    setupStr=setupStr,
                )
                print(f'error_stats["{targetSpecID}"] = \\')
                pprint(stats["errors"])
                errs = stats["errors"]
                for k, vals in errs.items():
                    data[k].extend(vals)
        
        with open(tmpfn, "wb") as f:
            pickle.dump((all_dfCompounds, data), f)
        print(f"Saved all_dfCompounds to '{tmpfn}'")
        
    if plot:
        # All setups, hue by setup
        plot_ar = 0.15
        plotPredictionDiagonals(all_dfCompounds, ar=plot_ar, show=show)
        for isoFit, incrFit in ((False, False), (False, True), (True, False)):
            # Single setups, hue by compound specific signal to noise
            plotPredictionDiagonals(
                all_dfCompounds, ar=plot_ar, setup=(isoFit, incrFit), show=show
            )

        if cfg.get("ar", None) is None:
            cfg["ar"] = arSpan
        ars = [float(ar) for ar in cfg["ar"]]
        figname = "quantitativeReconstructionErrors" + str(ars)
        plotErrorStats(data, figname, show=show)


if __name__ == "__main__":
    # Test run
    cfg = dict(
        lib="InHouse",
        mix="all.all",
        recompute=False,
        task="comparison_quantification",
        setup="all",
        # Assignment radii to compute (ar=None shows all)
        ar=None,
        plot=True,
        outdir=cache_dir,
    )

    main(cfg, show=True, loadtmp=False)
