import numpy as np
from pprint import pprint

from mcfnmr.demodata import case_dict
from mcfnmr.demodata.loading import (
    loadMMLibrary,
    loadMMMixtureLists,
    loadMMMixPointSpectra,
    loadMetaboMinerSpectrum,
)
from mcfnmr.demodata import mixNamesOLDB
from mcfnmr.routines.utils import (
    runMetaboMinerFits,
    prepareParameters,
    cache_dir,
    totalWeights,
    makeResultsDF,
    filterDF,
    collect_and_serialize_MM_grid_spectra,
)
from mcfnmr.utils.pointspectrum import makeRasterPointSpectrum
from mcfnmr.demodata import binning
from mcfnmr.config import DEBUG


def indicateContainment(df, th, containedCompounds, scale_by_peaknr):
    compounds = sorted(set(df["compound"]))
    assert len(compounds) == len(df["compound"])

    results = {}
    for c, assignment, npeaks in zip(df["compound"], df["assigned"], df["nPeaks"]):
        # If more flow than th is assigned to the compound in total, it is indicated as contained
        if th == 0.0:
            predicted = assignment > th
        elif scale_by_peaknr:
            predicted = False if npeaks == 0 else assignment >= th * npeaks
        else:
            predicted = assignment >= th
        # True containment
        contained = c in containedCompounds

        results[c] = (predicted, contained)
    
    for c in containedCompounds:
        if c in results:
            continue
        else:
            # c not contained in lib
            if DEBUG > 1:
                print(f"Compound '{c}' not in library, therefore it is not predicted.")
            results[c] = (False, True)
            pass
            
    return results


def classifyCompounds(
    dfCompounds, pars, th, containedCompounds, scale_by_peaknr, verb=False
):
    libID, targetID = pars["libID"], pars["targetSpecID"]
    targetName = mixNamesOLDB.get(targetID, targetID)

    filters = {"lib": libID, "target": targetName}
    dfC = filterDF(dfCompounds, filters)

    classificationResult = indicateContainment(
        dfC, th, containedCompounds, scale_by_peaknr
    )

    tp = [c for c, r in classificationResult.items() if r == (True, True)]
    tn = [c for c, r in classificationResult.items() if r == (False, False)]
    fp = [c for c, r in classificationResult.items() if r == (True, False)]
    fn = [c for c, r in classificationResult.items() if r == (False, True)]

    if verb:
        print("\nFN['%s'] = %s" % (targetID, str(fn)))
        print("FP['%s'] = %s" % (targetID, str(fp)))

    npos = len(tp) + len(fp)
    if npos:
        precision = len(tp) / npos
    else:
        precision = 0.0
    ntrue = len(tp) + len(fn)
    if ntrue:
        recall = len(tp) / ntrue
    else:
        recall = 1

    fscore = (
        precision * recall * 2 / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    res = dict(
        recall=recall,
        precision=precision,
        fscore=fscore,
        TP=tp,
        FP=fp,
        FN=fn,
        compound_df=dfC,
    )
    return res


def classificationStats(
    data, pars, th, mixSubs, mixIDs, scale_by_peaknr=False, verb=False
):
    stats = {}
    for mixID in mixIDs:
        dfCompounds = data[mixID]["compounds"]
        pars["targetSpecID"] = data[mixID]["targetSpecID"]

        if verb:
            print(f"\nMix '{mixID}'")

        if mixID[-4:] == "(dp)":
            mixID_stripped = mixID[:-4]
        else:
            mixID_stripped = mixID
        contained = mixSubs[mixID_stripped]

        cl = classifyCompounds(
            dfCompounds,
            pars,
            th,
            containedCompounds=contained,
            scale_by_peaknr=scale_by_peaknr,
            verb=verb,
        )

        if verb:
            print("\n%s:" % mixID)
            for k in cl:
                if k in ["FN", "FP", "TP"]:
                    print(f"'{k}': %d" % len(cl[k]))
                elif k == "compound_df":
                    pass
                else:
                    print(f"'{k}': %g" % cl[k])

        stats[mixID] = {"classificationThreshold": th, "classification": cl}

    return stats


def runSingleClassificationIncremental(caseSpec, pars, arSpan, th, load=True):
    # This calculates a series of incremental fits for the ar in arSpan
    # And gives the result for the maximal ar
    assert caseSpec["incrementalFit"]

    mixIDs = caseSpec["mixIDs"]
    mixSubs = loadMMMixtureLists()
    if pars["pointTarget"]:
        mixSpecs = loadMMMixPointSpectra()
    else:
        pars["binning"] = None
        mixSpecs = collect_and_serialize_MM_grid_spectra(mixIDs, pars)
    lib = loadMMLibrary(pars["libID"])

    pars["arSpan"] = arSpan
    results = runMetaboMinerFits(
        mixIDs=mixIDs,
        base_pars=pars,
        lib=lib,
        mixSpecs=mixSpecs,
        # pointTarget=caseSpec["pointTarget"],
        returnAllResults=True,
        load=load,
    )

    dfs = {}
    for mixID in mixIDs:
        result = results[mixID][max(arSpan)]
        dfTargets, dfCompounds = makeResultsDF(
            [{"res": result, "pars": pars}], originalWeights=totalWeights(lib)
        )
        dfs[mixID] = {
            "targets": dfTargets,
            "compounds": dfCompounds,
            "targetSpecID": pars["targetSpecID"],
            "targetTotalWeight": result.originalWeightY,
        }

    stats = {}
    for ar in arSpan:
        pars["assignmentRadius"] = ar
        stats[ar] = classificationStats(dfs, pars, th, mixSubs, mixIDs)

    print("Classification results:")
    pprint(stats[max(arSpan)])

    return stats


def runSingleClassification(caseSpec, pars, ar, th, load=True):
    assert not caseSpec["incrementalFit"]
    mixIDs = caseSpec["mixIDs"]
    mixSubs = loadMMMixtureLists()
    lib = loadMMLibrary(pars["libID"])

    if pars["pointTarget"]:
        mixSpecs = loadMMMixPointSpectra()
    else:
        pars["binning"] = None
        mixSpecs = collect_and_serialize_MM_grid_spectra(mixIDs, pars)

    pars["assignmentRadius"] = ar
    results = runMetaboMinerFits(
        base_pars=pars, lib=lib, mixSpecs=mixSpecs, mixIDs=mixIDs, load=load
    )
    dfs = {}
    for mixID in mixIDs:
        result = results[mixID]
        dfTargets, dfCompounds = makeResultsDF(
            [{"res": result, "pars": pars}], originalWeights=totalWeights(lib)
        )
        dfs[mixID] = {
            "targets": dfTargets,
            "compounds": dfCompounds,
            "targetSpecID": pars["targetSpecID"],
            "targetTotalWeight": result.originalWeightY,
        }
    scores = classificationStats(dfs, pars, th, mixSubs, mixIDs)

    print("Classification results:")
    pprint(scores)

    return scores


def main(cfg):
    assert cfg["task"] == "classification"
    assert cfg["lib"] == "MetaboMiner"
    assert cfg["mix"].split("_")[0] in ["N925", "N926", "N987", "N988"]
    assert cfg["setup"] in ["A", "B", "C", "D", "E", "F"]

    target_info = cfg["mix"].split("_")
    if len(target_info) > 1:
        smoothingRadius = float(target_info[2])
    else:
        smoothingRadius = 0.0
    mixID = target_info[0]

    # assignment radii (hard-coded for incremental approach)
    arSpan = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.175, 0.2, 0.25]
    arSpan = np.array(sorted(arSpan))

    setupSpec = case_dict[cfg["setup"]][mixID]
    incrementalFit = setupSpec["incrementalFit"]

    kwargs = dict(
        incrementalFit=incrementalFit,
        isolated_fit=setupSpec["isolated_fit"],
        noiseFilterLevel=None,
        pointTarget=setupSpec["pointTarget"],
        smoothing=smoothingRadius,
        workingDir=cfg["outdir"],
        recompute=cfg.get("recompute", True),
    )
    pars = prepareParameters(
        specDomain="metabominer", targetSpecID=cfg["mix"], libID=cfg["lib"], **kwargs
    )

    load = not pars["recompute"]
    if incrementalFit:
        ars = np.append(arSpan[arSpan < cfg["ar"]], cfg["ar"])
        _ = runSingleClassificationIncremental(
            setupSpec, pars, arSpan=ars, th=cfg["th"], load=load
        )
    else:
        _ = runSingleClassification(
            setupSpec, pars, ar=cfg["ar"], th=cfg["th"], load=load
        )


if __name__ == "__main__":
    # Test run
    cfg = dict(
        lib="MetaboMiner",
        recompute=False,
        task="classification",
        setup="A",
        mix="N925",
        # optimal for setup A
        ar=0.08,
        th=0.003,
        # mix="N926",
        # # optimal for N926
        # ar = 0.03,
        # th = 5e-05,
        outdir=cache_dir,
    )

    main(cfg)
