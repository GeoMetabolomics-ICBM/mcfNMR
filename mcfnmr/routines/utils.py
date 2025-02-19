import os
from pathlib import Path
from timeit import default_timer
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
from pprint import pp

from mcfnmr.config import default_pars, OUTDIR, DEBUG
from mcfnmr.core.mcf import mcf
from mcfnmr.demodata import binning
from mcfnmr.demodata import (
    mixNamesOLDB,
    compoundShortNames,
    scanNumbersMixes,
    scanNumberIndividualCompounds,
    intensityScalings,
    expectedConcentrationFactors,
)
from mcfnmr.demodata.loading import loadMetaboMinerSpectrum
from mcfnmr.utils.pointspectrum import makeRasterPointSpectrum

cache_dir = Path(OUTDIR) / "cache_demo_routines"
if not cache_dir.exists():
    os.mkdir(cache_dir)
    print(f"Created cache directory {cache_dir}")

figs_dir = OUTDIR / "figs"
figs_dir_oldb = figs_dir / "quantification"
figs_dir_MM = figs_dir / "classification MM"
if not figs_dir_oldb.exists():
    os.makedirs(figs_dir_oldb)
    print(f"Created directory {figs_dir_oldb}")
if not figs_dir_MM.exists():
    os.makedirs(figs_dir_MM)
    print(f"Created directory {figs_dir_MM}")


def makeTargetSpecification(pars, targetID, mixID=None):
    if mixID is None:
        mixID = targetID
    if not pars["pointTarget"]:
        nbin = pars.get("binning")
        if nbin is None:
            nbin = binning.get(mixID, None)
            if nbin is None:
                print("No nbin for", mixID)
        smoothstr = ", smooth%g" % pars["smoothing"] if pars["smoothing"] else ""
        binstr = ", (%dx%d)-bins" % (nbin)
        targetSpecification = targetID + "(raster%s%s)" % (smoothstr, binstr)
    else:
        targetSpecification = targetID + "(peaklist)"
    return targetSpecification


def makeMCFResultFilename(pars, mixID):
    if "specificTargetSpecID" not in pars:
        pars["specificTargetSpecID"] = makeTargetSpecification(
            pars, pars["targetSpecID"], mixID
        )
    targetID = pars["specificTargetSpecID"]
    compoundFit = "isolated" if pars["isolated_fit"] else "joint"
    noiseFilterLevelStr = (
        "_noiseFilter%g" % pars["noiseFilterLevel"] if pars["noiseFilterLevel"] else ""
    )
    ar = pars["assignmentRadius"]
    if "previousARSteps" in pars:
        prevArSteps = ["%g" % s for s in sorted(pars["previousARSteps"])]
        arStr = f"_%gon%s" % (float(ar), "[" + (",".join(prevArSteps)) + "]")
    else:
        arStr = "_ar%g" % ar
    libID = pars["libID"]
    fn = (
        f"mcfResult_{compoundFit}_in_{targetID}_lib_{libID}{noiseFilterLevelStr}{arStr}"
    )
    fn = fn + ".pickle"
    return fn


def collect_and_serialize_MM_grid_spectra(mixIDs, pars):
    mixSpecs = dict()
    for mixID in mixIDs:
        targetRasterSpec = loadMetaboMinerSpectrum(
            mixID, denoising_pars={"smoothRadius": pars["smoothing"]}
        )
        nbins = pars.get("binning", None)
        if nbins is None:
            nbins = binning[mixID]
        mixSpecs[mixID] = makeRasterPointSpectrum(
            targetRasterSpec,
            nbin=nbins,
            signalThreshold=pars["noiseFilterLevel"],
        )
    return mixSpecs


def runMetaboMinerFits(
    mixIDs, base_pars, lib, mixSpecs, returnAllResults=False, load=True, verb=1
):
    """
    Fit Metabominer data and compute precision/recall/F score
    """
    incrementalFit = base_pars["incrementalFit"]
    pointTarget = base_pars["pointTarget"]

    results = {}
    for mixID in mixIDs:
        pars = deepcopy(base_pars)  # protect from mutation
        pars["targetSpecID"] = mixID
        results[mixID] = {}
        targetPointSpec = mixSpecs[mixID]
        if pointTarget:
            # working with point spectra as targets
            # load regions from raster spectrum
            if mixID[-4:] == "(dp)":
                # Deeppicker variant of peaklist
                targetRasterSpec = loadMetaboMinerSpectrum(
                    mixID[:-4], denoising_pars={"smoothRadius": pars["smoothing"]}
                )
            else:
                targetRasterSpec = loadMetaboMinerSpectrum(
                    mixID, denoising_pars={"smoothRadius": pars["smoothing"]}
                )

            pars["binning"] = (1, 1)
            targetPointSpec.regions = targetRasterSpec.regions
        else:
            print(
                f"runMetaboMinerFits(): Binning for mix {mixID}: {targetPointSpec.binning}"
            )

        pars["specificTargetSpecID"] = makeTargetSpecification(
            pars, pars["targetSpecID"], mixID
        )

        # Generate or load MCF results
        if incrementalFit:
            # Initial setup for AR steps.
            # This triggers incremental fit in singleRun()
            pars["previousARSteps"] = []
            pars["reservedSinkCapacity"] = np.zeros_like(targetPointSpec.weights)
            prevResult, prevSavefn = None, None
            for ar in sorted(pars["arSpan"]):
                pars["assignmentRadius"] = ar
                savefn = cache_dir / makeMCFResultFilename(pars, mixID)
                result = singleRun(
                    pars, lib, targetPointSpec, load=load, verb=verb, savefn=savefn
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
                results[mixID][ar] = result
                pars["previousARSteps"].append(float(ar))
                pars["reservedSinkCapacity"] = getattr(
                    result, "reservedSinkCapacity", None
                )
                prevResult = result
        else:
            savefn = cache_dir / makeMCFResultFilename(pars, mixID)
            results[mixID] = singleRun(
                pars, lib, targetPointSpec, load=load, verb=verb, savefn=savefn
            )

    if not returnAllResults and incrementalFit:
        results = {
            mixID: results_i[max(results_i.keys())]
            for mixID, results_i in results.items()
        }
    return results


def updateSavedMCFResult(result, savefn):
    with open(savefn, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved updated result to '{savefn}'")


def prepareParameters(specDomain, targetSpecID, libID, **kwargs):
    pars = deepcopy(default_pars)

    # Whether to use the peak list dat of the target (or the raster data)
    # (Only effective if both are available)
    assert "pointTarget" in pars

    pars["specDomain"] = specDomain
    pars["targetSpecID"] = targetSpecID
    pars["libID"] = libID

    pars.update(kwargs)
    return pars


def totalWeights(lib):
    weights = {}
    for c, spec in lib.items():
        w = sum(spec.weights)
        weights[c] = w
    return weights


def incrementalMCFResultsUpdate(res, inflowsY, prev):
    if prev is None:
        # res must be the first result in a sequence of
        # assignment radii if we are here
        res.reservedSinkCapacity = inflowsY
        res.totalReserved = np.sum(res.reservedSinkCapacity)
        return

    res.reservedSinkCapacity = prev.reservedSinkCapacity + inflowsY
    res.totalReserved = np.sum(res.reservedSinkCapacity)

    res.originalWeightY = prev.originalWeightY

    totalAbsorbed = res.totalAbsorption + prev.totalAbsorption
    absorptionCost = (
        res.totalAbsorption * res.absorptionCost
        + prev.totalAbsorption * prev.absorptionCost
    ) / totalAbsorbed
    res.totalAbsorbed = totalAbsorbed
    res.absorptionCost = absorptionCost

    assigned = res.assigned + prev.assigned
    assignedIx = assigned > 0
    assignmentCosts = (
        res.assigned * res.assignmentCosts + prev.assigned * prev.assignmentCosts
    )
    assignmentCosts[assignedIx] /= assigned[assignedIx]
    res.assigned = assigned
    res.assignmentCosts = assignmentCosts
    res.specificCosts[assignedIx] = assignmentCosts[assignedIx] / assigned[assignedIx]

    res.inflowsC += prev.inflowsC
    res.totalAbsorption += prev.totalAbsorption
    res.totalAssignedFlow += prev.totalAssignedFlow

    res.totalAbsorptionCost = res.totalAbsorption * res.absorptionCost
    res.totalAssignmentCost = sum(res.assignmentCosts)
    res.totalCost = res.totalAbsorptionCost + res.totalAssignmentCost


def singleRun(pars, lib, targetPointSpec, savefn, verb=1, load=True):
    pars = deepcopy(pars)
    targetPointSpec = deepcopy(targetPointSpec)
    targetPointSpec.weights = np.maximum(targetPointSpec.weights, 0.0)

    # This indicates, that this fit is a step in a assignment
    # radius increase / sink reservation series
    incrementalFit = "previousARSteps" in pars
    if incrementalFit and not (load and savefn.exists()):
        prevSteps = pars["previousARSteps"]
        if len(prevSteps) > 0:
            assert "reservedSinkCapacity" in pars
            # We have previous assignments → reserve the sinks
            targetPointSpec.weights -= pars["reservedSinkCapacity"]
            reserved_precision_tolerance = 1e-16 * max(targetPointSpec.weights)
            if min(targetPointSpec.weights) < 0.0:
                if min(targetPointSpec.weights < -reserved_precision_tolerance):
                    # Negativity exceeds tolerated round-off errors
                    raise Exception(
                        "Incremental fit encountered reserved capacity exceeding target capacity."
                    )
                else:
                    # Ignore negativity, since it is probably due to a rounding error
                    targetPointSpec.weights = np.maximum(targetPointSpec.weights, 0.0)

        if DEBUG > 0:
            rc = pars.get("reservedSinkCapacity", [0.0])
            print(
                "\nIncremental fit:\n   total reserved capacity: %g\n    min/max: %g/%g\n"
                % (sum(rc), min(rc), max(rc))
            )

    resolveYinResult = incrementalFit

    tic = default_timer()
    mcfResult = mcf(
        target_spectrum=targetPointSpec,
        library=lib,
        assignment_radius=pars["assignmentRadius"],
        absorption_cost=pars["absorptionCost"],
        target_id=pars["specificTargetSpecID"],
        lib_id=pars["libID"],
        target_regions=targetPointSpec.regions,
        isolated_fit=pars["isolated_fit"],
        resolveYinResult=resolveYinResult,
        load_dists=pars.get("load_dists", False),
        load=load,
        savefn=savefn,
    )
    print("Total time for MCF: %g secs\n" % (default_timer() - tic))
    return mcfResult


def filterDF(dfInit, filters):
    df = deepcopy(dfInit)
    for k, v in filters.items():
        df = df.loc[df[k] == v]
    return df


def makeResultsDF(data, originalWeights, intensityCorrection=True):
    # NOTE: Consider splitting this for purposes of MetaboMiner and OLDB-experiments
    # (handling both cases makes this difficult to understand)
    mcfResults = {}
    params = {}
    for d in data:
        res, pars = d["res"], d["pars"]
        fn = makeMCFResultFilename(pars, pars["targetSpecID"])
        if DEBUG > 0:
            if fn in mcfResults:
                raise Exception("makeResultsDF(): Duplicate run id '%s'" % fn)
        mcfResults[fn] = res
        params[fn] = pars

    runIDs = sorted(mcfResults.keys())

    dfCompounds = {
        "compound": [],
        "target": [],
        "lib": [],
        "assignment radius": [],
        "assigned": [],
        "absorbed": [],
        "specific cost": [],
        "concentration factor": [],
        "absorption factor": [],
        "conc.fctr. mismatch": [],
        "isolated fit": [],
        "incremental fit": [],
        "nPeaks": [],
        "intensity scaling": [],  # automatic intensity scaling
        "expected concentration factor": [],
    }
    dfTargets = {
        "target": [],
        "lib": [],
        "assignment radius": [],
        "assigned": [],
        "absorbed": [],
        "specific cost": [],
        "isolated fit": [],
        "originalWeightY": [],
    }

    for runID in runIDs:
        pars, results = params[runID], mcfResults[runID]
        targetID = pars["targetSpecID"]
        if targetID in mixNamesOLDB:
            targetName = mixNamesOLDB[targetID]
        elif targetID.split("_")[0] in compoundShortNames:
            compoundFullName = targetID.split("_")[0]
            targetName = compoundShortNames[compoundFullName]
        else:
            targetName = targetID
        if targetID in scanNumbersMixes:
            # Corrector for the mix concentration (all individual compounds are created by 8 scans,
            # mixes sometimes at differing numbers)
            nScansY = scanNumbersMixes[targetID]
        else:
            nScansY = scanNumberIndividualCompounds
        libID = pars["libID"]
        ar = pars["assignmentRadius"]
        isolated_fit = pars["isolated_fit"]
        incrFit = pars["incrementalFit"]

        if type(results) is not list:
            # Transform single result to data structure obtained if no absorption limits are specified
            results = [{"res": results, "absorptionLimit": None}]

        for resInfo in results:
            res = resInfo["res"]
            compounds = sorted(res.compounds.keys())
            if DEBUG > 0:
                missing = [c for c in compounds if not c in originalWeights]
                if missing:
                    print(
                        f"Compounds {missing}: no original weight found -> not in lib?"
                    )
                    print(libID)

            if DEBUG > 1:
                print("originalWeights (%d):" % len(originalWeights))
                pp(originalWeights)
                print("\n ################ \n\ncompounds (%d):" % len(compounds))
                print(compounds)
                print(f"ar: {ar}")
                print(f"libID: {libID}")
                targetSpecID = pars["targetSpecID"]
                print(f"mixID: {targetSpecID}")

            # this ensures that the total assigned flow < 1.0
            scale = 1 / res.originalWeightY

            assigned = dict(zip(compounds, res.assigned * scale))
            specificCosts = dict(zip(compounds, res.specificCosts))
            absorbed = dict(list(zip(compounds, np.zeros_like(compounds, dtype=float))))
            for i, c in enumerate(compounds):
                if getattr(res, "nPeaksConsidered", None) is None:
                    nPeaks = len(res.compounds[c]["ix"])
                else:
                    nPeaks = res.nPeaksConsidered[i]
                dfCompounds["compound"].append(compoundShortNames.get(c, c))
                dfCompounds["target"].append(targetName)
                dfCompounds["assignment radius"].append(ar)
                dfCompounds["assigned"].append(assigned[c])
                dfCompounds["absorbed"].append(absorbed[c])
                dfCompounds["specific cost"].append(specificCosts[c])
                intScaling = intensityScalings.get(c, None)
                dfCompounds["intensity scaling"].append(intScaling)
                expectedConcentrationFactor = expectedConcentrationFactors.get(
                    targetID, {}
                ).get(c, None)
                dfCompounds["expected concentration factor"].append(
                    expectedConcentrationFactor
                )

                # The concentration factor is calculated as the absolute
                # assigned flow per scan number related to the original
                # compound spectrum intensity per scan number
                cf = (
                    assigned[c]
                    * scanNumberIndividualCompounds
                    / (nScansY * originalWeights[c] * scale)
                )
                af = (
                    absorbed[c]
                    * scanNumberIndividualCompounds
                    / (nScansY * originalWeights[c] * scale)
                )
                if intensityCorrection and c in intensityScalings:
                    cf *= 2 ** (intensityScalings[targetID] - intensityScalings[c])
                    af *= 2 ** (intensityScalings[targetID] - intensityScalings[c])
                else:
                    af = cf = None
                dfCompounds["concentration factor"].append(cf)
                dfCompounds["absorption factor"].append(af)
                if expectedConcentrationFactor:
                    dcf = cf - expectedConcentrationFactor
                    dfCompounds["conc.fctr. mismatch"].append(dcf)
                else:
                    dfCompounds["conc.fctr. mismatch"].append(None)
                dfCompounds["isolated fit"].append(isolated_fit)
                dfCompounds["incremental fit"].append(incrFit)
                dfCompounds["lib"].append(libID)
                dfCompounds["nPeaks"].append(nPeaks)
            totalAbsorbed = res.totalAbsorption * scale
            totalAssigned = res.totalAssignedFlow * scale
            if totalAssigned > 0:
                totalSpecificCost = (
                    sum([specificCosts[c] * assigned[c] for c in compounds])
                    / totalAssigned
                )
            else:
                totalSpecificCost = None
            dfTargets["target"].append(targetName)
            dfTargets["assignment radius"].append(ar)
            dfTargets["assigned"].append(totalAssigned)
            dfTargets["absorbed"].append(totalAbsorbed)
            dfTargets["specific cost"].append(totalSpecificCost)
            dfTargets["isolated fit"].append(isolated_fit)
            dfTargets["lib"].append(libID)
            dfTargets["originalWeightY"].append(res.originalWeightY)

    dfTargets["assigned"] = [float(x) for x in dfTargets["assigned"]]
    dfTargets = pd.DataFrame(dfTargets)
    dfCompounds = pd.DataFrame(dfCompounds)
    dfCompounds["assigned"] = [float(x) for x in dfCompounds["assigned"]]
    dfCompounds["compound"] = [str(x) for x in dfCompounds["compound"]]
    return dfTargets, dfCompounds
