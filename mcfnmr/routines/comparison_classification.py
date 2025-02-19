import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from mcfnmr.demodata import caseSpecC, caseSpecB, caseSpecA, caseSpecD
from mcfnmr.demodata.loading import (
    loadMMMixtureLists,
    loadMMMixPointSpectra,
    loadMMLibrary,
    loadSMARTMINERBenchmarks,
)
from mcfnmr.routines.classification import classificationStats
from mcfnmr.routines.utils import (
    cache_dir,
    runMetaboMinerFits,
    prepareParameters,
    figs_dir,
    totalWeights,
    makeResultsDF,
    makeTargetSpecification,
    collect_and_serialize_MM_grid_spectra,
)
from mcfnmr.demodata import mix_selections
from mcfnmr.config import DEBUG


def plotMetaboMinerComparison(
    libID,
    df,
    plotOnlyFScore=False,
    title=None,
    show=True,
    show_legend=True,
    plot_width=2.5,
    plot_height=None,
    pal=None,
):
    if libID is not None:
        df = df[df["libID"] == libID]
    if title is None:
        title = f"Comparison MetaboMiner-EMD lib '{libID}'"
    if pal is None:
        pal = sns.color_palette("magma", n_colors=7)
    nrows = 1 if plotOnlyFScore else 3
    plt.style.use("bmh")
    if plot_height:
        figsize = (plot_width, plot_height)
    else:
        figsize = (plot_width, 2 * nrows)
    fig, axes = plt.subplots(
        figsize=figsize, nrows=nrows, sharex=True, layout="constrained"
    )
    if plotOnlyFScore:
        axes = [axes]
    variables = ("F1 score", "precision", "recall")
    fig.suptitle(title)
    for ax, var in zip(axes, variables):
        ax.set_xlim((0.0, 1.0))
        sns.barplot(df, y="mixID", x=var, hue="method", palette=pal, ax=ax)

    if DEBUG > 0:
        dfg = df.groupby("method")
        for method in dfg.groups:
            print("Method:", method)
            g = dfg.get_group(method)
            print("mixes:", list(g["mixID"]))
            print("avg. fscore:", np.mean(g["F1 score"]))
            print("avg. recall:", np.mean(g["recall"]))
            print("avg. precision:", np.mean(g["precision"]))
            gg = g.groupby("mixID")
            for mix in gg.groups:
                dfgg = gg.get_group(mix)
                print("   mixID:", mix)
                print("     fscore:", list(dfgg["F1 score"]))
                print("     recall:", list(dfgg["recall"]))
                print("     precision:", list(dfgg["precision"]))
                if mix == "all":
                    nFN = nFP = nTP = None
                else:
                    nTP = dfgg['TP'].iloc[0]
                    nTP = len(nTP) if type(nTP) is list else nTP
                    nFP = dfgg['FP'].iloc[0]
                    nFP = len(nFP) if type(nFP) is list else nFP
                    nFN = dfgg['FN'].iloc[0]
                    nFN = len(nFN) if type(nFN) is list else nFN
                print(f"     TP ({nTP}):", list(dfgg["TP"]))
                print(f"     FN ({nFN}):", list(dfgg["FN"]))
                print(f"     FP ({nFP}):", list(dfgg["FP"]))

    if not show_legend:
        ax.legend(loc=2).remove()
    fn = figs_dir / f"{title}_fscores.svg"
    fig.savefig(fn)
    print(f"\nSaved figure '{fn}'")

    if show:
        plt.show()
    else:
        plt.close(fig)


def makeMethodName(caseSpec):
    name = "MCF (%s)" % caseSpec["name"]
    return name


def evaluateMCFPerformance(pars, caseSpec, assignmentRadius, th, load=True):
    pars["isolated_fit"] = caseSpec["isolated_fit"]
    pars["assignmentRadius"] = assignmentRadius
    pars["libID"] = caseSpec["libID"]

    mixIDs = caseSpec["mixIDs"]
    mixSubs = loadMMMixtureLists()
    lib = loadMMLibrary(pars["libID"])

    if pars["pointTarget"]:
        mixSpecs = loadMMMixPointSpectra()
    else:
        pars["binning"] = None
        mixSpecs = collect_and_serialize_MM_grid_spectra(mixIDs, pars)

    results = runMetaboMinerFits(
        mixIDs=mixIDs, base_pars=pars, lib=lib, mixSpecs=mixSpecs, load=load
    )

    dfs = {}
    for mixID in mixIDs:
        result = results[mixID]
        if pars["pointTarget"]:
            pars["targetSpecID"] = mixID + "(peaklist)"
        else:
            pars["targetSpecID"] = mixID
            pars["specificTargetSpecID"] = makeTargetSpecification(
                pars, pars["targetSpecID"], mixID
            )
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

    df = {
        "mixID": [],
        "libID": [],
        "assignmentRadius": [],
        "binning": [],
        "absorptionCost": [],
        "precision": [],
        "recall": [],
        "F1 score": [],
        "cutoffFlag": [],
        "hueID": [],
        "method": [],
        "FP": [],
        "TP": [],
        "FN": [],
    }

    method = makeMethodName(caseSpec)
    for mixID, data in scores.items():
        df["mixID"].append(mixID)
        df["precision"].append(data["classification"]["precision"])
        df["recall"].append(data["classification"]["recall"])
        df["F1 score"].append(data["classification"]["fscore"])
        df["FP"].append(data["classification"]["FP"])
        df["TP"].append(data["classification"]["TP"])
        df["FN"].append(data["classification"]["FN"])
        df["libID"].append(caseSpec["libID"])
        df["assignmentRadius"].append(assignmentRadius)
        df["absorptionCost"].append(pars["absorptionCost"])
        df["binning"].append(pars["binning"])
        df["method"].append(method)

    for k in list(df.keys()):
        if len(df[k]) == 0:
            df[k] = None
    df = pd.DataFrame(df)
    return df


def main(cfg, show=False):
    wdir = cache_dir / "method_comparison"
    assert cfg["task"] == "comparison_classification"
    assert cfg["lib"] == "MetaboMiner"
    assert cfg["mix"] in ["N925", "N987", "N988", "all"]

    kwargs = dict(
        normalizeY=False,
        specDomain="MetaboMiner",
        incrementalFit=False,
        noiseFilterLevel=None,
        smoothing=0,
        workingDir=cfg["outdir"],
        recompute=cfg.get("recompute", False),
    )
    pars = prepareParameters(targetSpecID=cfg["mix"], libID=cfg["lib"], **kwargs)

    if cfg["mix"] == "all":
        mixIDs = mix_selections["SMART-Miner"]
    else:
        mixIDs = [cfg["mix"]]

    # assignment radii (hard-coded for incremental approach)
    arSpan = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.175, 0.2, 0.25]

    includeAverageOverAllMixes = True
    # Select mix selection, assignment radius and detection threshold
    mixSelection = "SMART-Miner"

    # Hard-coded optimal assignment radii and detection thresholds for comparison
    optPars = {
        "setupA": {"ar": 0.06, "th": 0.003},
        "setupB": {"ar": 0.05, "th": 0.003},
        "setupC": {"ar": 0.08, "th": 0.003},
        "setupD": {"ar": 0.06, "th": 0.005},
    }

    caseSpecIncr = deepcopy(caseSpecC)
    testsets = [
        {"case": caseSpecA, **optPars["setupA"]},  # "standard"
        {"case": caseSpecB, **optPars["setupB"]},  # "independent"
        {"case": caseSpecIncr, **optPars["setupC"]},  # "incremental"
        {"case": caseSpecD, **optPars["setupD"]},  # "peak list"
    ]

    libID = "Biofluid ( all )"
    for v in testsets:
        v["case"]["libID"] = libID

    # For incremental fit, prepare previousARSteps to allow loading
    # previously computed incremental fit result
    # → set load=True for evaluateMCFPerformance()
    previousARSteps = [x for x in arSpan if x < optPars["setupC"]["ar"]]
    previousAssignments = dict()
    for mixID in caseSpecIncr["mixIDs"]:
        previousAssignments[mixID] = dict(previousARSteps=deepcopy(previousARSteps))
    caseSpecIncr["previousAssignments"] = previousAssignments

    # Generate data for selected mixes
    dfMCFall = []
    for d in testsets:
        print("\n#  ", d["case"]["name"])
        arSpan_restricted = [float(ar) for ar in arSpan if ar <= d["ar"]]
        assert d["ar"] == arSpan_restricted[-1]
        kwargs = dict(
            arSpan=arSpan_restricted,
            incrementalFit=d["case"]["incrementalFit"],
            isolated_fit=d["case"]["isolated_fit"],
            noiseFilterLevel=None,
            pointTarget=d["case"]["pointTarget"],
            smoothing=0,
            workingDir=wdir,
            recompute=False,
        )
        pars = prepareParameters(
            specDomain="metabominer", targetSpecID=None, libID=None, **kwargs
        )
        dfMCFall.append(
            evaluateMCFPerformance(
                pars, d["case"], assignmentRadius=d["ar"], th=d["th"], load=True
            )
        )
    dfSM = loadSMARTMINERBenchmarks()
    dfAll = pd.concat(dfMCFall + [dfSM])

    # Filter for selected mixes
    ix = [
        (mid in mixIDs) and lid == libID
        for mid, lid in zip(dfAll["mixID"], dfAll["libID"])
    ]
    dfAll = dfAll.iloc[ix]

    if includeAverageOverAllMixes:
        # include the average F1 (and other numeric columns) over all mixes in dfAll
        numeric_cols = ["precision", "recall", "F1 score"]
        dfavg = {c: [] for c in dfAll.columns}
        dfg = dfAll.groupby("method")
        for method in dfg.groups:
            for k in dfAll.columns:
                if k == "method":
                    dfavg[k].append(method)
                elif k in numeric_cols:
                    dfm = dfg.get_group(method)
                    dfavg[k].append(np.mean(dfm[k]))
                elif k == "mixID":
                    dfavg[k].append("all")
                elif k == "libID":
                    dfavg[k].append(libID)
                else:
                    dfavg[k].append(None)

        dfavg = pd.DataFrame(dfavg)
        dfAll = pd.concat([dfAll, dfavg])

    # Compare full library against other algorithms
    ars = ["%g" % t["ar"] for t in testsets]
    ths = ["%g" % t["th"] for t in testsets]

    title = (
        f"Comparison MetaboMiner-MCF with lib '{libID}', mixes: '{mixSelection}', ar: %s, th: %s"
        % (ths, ars)
    )
    plotMetaboMinerComparison(
        libID=libID, df=dfAll, title=title, plotOnlyFScore=True, show=show
    )


if __name__ == "__main__":
    # Test run
    cfg = dict(
        lib="MetaboMiner",
        mix="all",
        recompute=False,
        task="comparison_classification",
        ar=dict(
            A=0.06,
            B=0.05,
            C=0.08,
            D=0.06,
        ),
        th=dict(
            A=0.003,
            B=0.003,
            C=0.003,
            D=0.005,
        ),
        outdir=cache_dir,
    )

    main(cfg, show=True)
