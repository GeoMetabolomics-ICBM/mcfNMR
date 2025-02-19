import numpy as np
from mcfnmr.demodata import case_dict
from mcfnmr.routines.utils import (
    prepareParameters,
    runMetaboMinerFits,
    cache_dir,
    makeResultsDF,
    totalWeights,
    collect_and_serialize_MM_grid_spectra,
)
from mcfnmr.routines.classification import classificationStats
from mcfnmr.demodata.loading import (
    loadMMMixtureLists,
    loadMMMixPointSpectra,
    loadMMLibrary,
)
from mcfnmr.routines.plotting import plotPerformanceScanResults, plotTimings
from mcfnmr.demodata import mix_selections


def makeScanTitle(caseSpec, smoothingRadius=0.0):
    mixIDs = caseSpec["mixIDs"]
    ptTargetStr = "pt" if caseSpec["pointTarget"] else "rst"
    if ptTargetStr == "rst" and smoothingRadius > 0:
        ptTargetStr += ", smooth%g" % smoothingRadius
    isolatedFitStr = "isofit" if caseSpec["isolated_fit"] else "jointfit"
    incrStr = ", incrFit" if caseSpec["incrementalFit"] else ""
    libID = caseSpec["libID"]
    fmttp = (str(mixIDs), ptTargetStr, incrStr, libID, isolatedFitStr)
    title = "mixes%s(%s)%s, lib %s(%s)" % fmttp
    return title


def runPerformanceScan(caseSpec, pars, thSpan, load=True):
    mixSubs = loadMMMixtureLists()
    lib = loadMMLibrary(pars["libID"])
    mixIDs = caseSpec["mixIDs"]
    if mixIDs == "all":
        mixIDs = mix_selections["SMART-Miner"]
    incrementalFit = caseSpec["incrementalFit"]

    if pars["pointTarget"]:
        mixSpecs = loadMMMixPointSpectra()
    else:
        pars["binning"] = None
        mixSpecs = collect_and_serialize_MM_grid_spectra(mixIDs, pars)

    if incrementalFit:
        results = runMetaboMinerFits(
            base_pars=pars,
            lib=lib,
            mixSpecs=mixSpecs,
            mixIDs=mixIDs,
            load=load,
            returnAllResults=True,
        )
        # Swap indexing order
        results = {
            ar: {mixID: results[mixID][ar] for mixID in mixIDs} for ar in pars["arSpan"]
        }

    else:
        results = dict()
        for ar in pars["arSpan"]:
            results[ar] = ({},)
            pars["assignmentRadius"] = ar
            results[ar] = runMetaboMinerFits(
                base_pars=pars, lib=lib, mixSpecs=mixSpecs, mixIDs=mixIDs, load=load
            )

    # Debug: timings plot
    # setup = caseSpec["name"]
    # plotTimings(results, setup, mixIDs, loglog=False, show=True)

    scanResults = {}
    for ar in pars["arSpan"]:
        dfs = {}
        scanResults[ar] = {}
        for mixID in mixIDs:
            result = results[ar][mixID]
            pars["targetSpecID"] = mixID
            dfTargets, dfCompounds = makeResultsDF(
                [{"res": result, "pars": pars}], originalWeights=totalWeights(lib)
            )
            dfs[mixID] = {
                "targets": dfTargets,
                "compounds": dfCompounds,
                "targetSpecID": pars["targetSpecID"],
                "targetTotalWeight": result.originalWeightY,
            }
        for th in thSpan:
            scores = classificationStats(dfs, pars, th, mixSubs, mixIDs, verb=False)
            scanResults[ar][th] = scores
    return scanResults


def main(cfg, show=False):
    assert cfg["task"] == "scan"
    assert cfg["lib"] == "MetaboMiner"
    assert cfg["mix"].split("_")[0] in ["N925", "N926", "N987", "N988", "N907", "all"]
    assert cfg["setup"] in ["A", "B", "C", "D", "E", "F"]

    setup = cfg["setup"]
    target_info = cfg["mix"].split("_")
    if len(target_info) > 1:
        smoothingRadius = float(target_info[2])
    else:
        smoothingRadius = 0.0
    mixID = target_info[0]

    setupSpec = case_dict[setup][mixID]
    incrementalFit = setupSpec["incrementalFit"]

    kwargs = dict(
        incrementalFit=incrementalFit,
        isolated_fit=setupSpec["isolated_fit"],
        noiseFilterLevel=None,
        pointTarget=setupSpec["pointTarget"],
        smoothing=smoothingRadius,
        workingDir=cfg["outdir"],
        recompute=cfg.get("recompute", True),
        # assignment radii to be scanned
        arSpan=np.array(sorted(cfg["ar"])),
    )

    pars = prepareParameters(
        specDomain="metabominer", targetSpecID=cfg["mix"], libID=cfg["lib"], **kwargs
    )

    # Scan performance in dependence of assignment radius  and detection threshold
    thSpan = np.linspace(*cfg["th"], cfg["nth"])
    scanResults = runPerformanceScan(
        setupSpec, pars, thSpan, load=not pars["recompute"]
    )
    axDict = plotPerformanceScanResults(
        scanResults,
        title=makeScanTitle(setupSpec, smoothingRadius),
        logAR=False,
        show=show,
    )


if __name__ == "__main__":
    # Test run
    cfg = dict(
        lib="MetaboMiner",
        # mix="N926_smooth_0.3",
        # mix="N925",
        mix="all",
        recompute=False,
        task="scan",
        setup="D",
        ar=[0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15],
        # ar=[0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.175, 0.2, 0.25],
        th=[0.0, 0.0015],
        nth=31,
        plot=True,
        outdir=cache_dir,
    )
    main(cfg, show=True)
