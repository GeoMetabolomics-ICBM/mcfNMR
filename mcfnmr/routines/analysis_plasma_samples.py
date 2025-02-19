import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pp
import pickle
from collections import defaultdict

from mcfnmr.demodata.loading import (
    loadMetaboMinerSpectrum,
    loadMMLibrary,
    loadMMMixtureLists,
    loadMMMixPointSpectra,
    buildHMDBID2NameMap,
    buildName2HMDBIDMap,
)
from mcfnmr.routines.utils import (
    prepareParameters,
    cache_dir,
    makeResultsDF,
    totalWeights,
    figs_dir,
    runMetaboMinerFits,
    collect_and_serialize_MM_grid_spectra,
)
from mcfnmr.utils.pointspectrum import makeRasterPointSpectrum, makeMix
from mcfnmr.demodata import case_dict
from mcfnmr.routines.classification import classificationStats
from mcfnmr.utils.plotting import plotLibOnTarget, plotRasterSpectrum, sqrt_transfo
from mcfnmr.demodata import binning, metaboMinerLibNames
from mcfnmr.routines.plotting import plotPerformanceScanResults, plotSlices
from mcfnmr.utils.rasterspectrum import sampleBlankSpectrum
from mcfnmr.routines.comparison_classification import plotMetaboMinerComparison
from mcfnmr.config import DATADIR, METABOMINER_DIR, SMARTMINER_DATADIR
from data.MetaboMiner.mixture_contents.map_to_HMDB import findHMDB_ID
from mcfnmr.utils.system import get_mcfnmr_home


outdir = figs_dir / "MM_N926"
if not outdir.exists():
    os.makedirs(outdir)
    print(f"Created directory '{outdir}'")


def makePerformanceTableCOLMAR(mixID, match_cutoff, peaks="deeppicker"):
    assert peaks in ["deeppicker", "metabominer"]

    COLMARDIR = DATADIR / "COLMAR"
    if peaks == "deeppicker":
        results = pd.read_csv(
            COLMARDIR / f"compound_report_{mixID}(dp)_0.04_COLMARm_refd.txt", sep=" "
        )
    else:
        results = pd.read_csv(
            COLMARDIR / f"compound_report_{mixID}_0.03_HSQC_refd.txt", sep=" "
        )

    mixSubs = loadMMMixtureLists()[mixID]

    df = {
        "libID": [],
        "F1 score": [],
        "precision": [],
        "recall": [],
        "method": [],
        "mixID": [],
        "ar_opt": [],
        "th_opt": [],
        "TP": [],
        "FN": [],
        "FP": [],
    }

    map_name2HMDB = buildName2HMDBIDMap(verb=0)
    all_names = sorted(map_name2HMDB.keys())
    detections = set()
    for name, match_ratio in zip(results["Name"], results["Matching_ratio"]):
        HMDB_ID = findHMDB_ID(name, map_name2HMDB, all_names)
        print(f"{name} → {HMDB_ID}")
        if HMDB_ID in detections:
            print(f"    → {HMDB_ID} ({name}) repeated.")
        if match_ratio >= match_cutoff:
            detections.add(HMDB_ID)
    df["libID"].append("COLMAR")
    if peaks == "deeppicker":
        df["method"].append(f"COLMARm (dp)")
    else:
        df["method"].append(f"COLMAR-HSQC")
    df["mixID"].append(mixID)
    df["ar_opt"].append(None)
    df["th_opt"].append(None)
    TP = detections.intersection(mixSubs)
    FP = detections.difference(mixSubs)
    FN = set(mixSubs).difference(detections)
    df["TP"].append(len(TP))
    df["FP"].append(len(FP))
    df["FN"].append(len(FN))
    recall = len(TP) / (len(TP) + len(FN))
    precision = len(TP) / (len(TP) + len(FP))
    fscore = (
        precision * recall * 2 / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    df["recall"].append(recall)
    df["precision"].append(precision)
    df["F1 score"].append(fscore)

    return pd.DataFrame(df), dict(TP=TP, FP=FP, FN=FN)


def makePerformanceTableMM(mixID, libID = "MetaboMiner-all"):
    table1 = pd.read_csv(
        METABOMINER_DIR / "tables_Xia_etal_2008" / "table1.txt", sep=" "
    )
    # table2 = pd.read_csv(METABOMINER_DIR / "tables_Xia_etal_2008" / "table2.txt", sep=" ")
    # table3 = pd.read_csv(METABOMINER_DIR / "tables_Xia_etal_2008" / "table3.txt", sep=" ")
    table_N907 = pd.read_csv(
        METABOMINER_DIR / "tables_Xia_etal_2008" / "table_N907.txt", sep=" "
    )
    
    table1 = pd.concat((table1, table_N907))
    table1.index = list(range(len(table1)))
    
    assert "N880" != mixID  # No performance provided for full lib

    df = {
        "libID": [],
        "F1 score": [],
        "precision": [],
        "recall": [],
        "method": [],
        "mixID": [],
        "ar_opt": [],
        "th_opt": [],
        "TP": [],
        "FN": [],
        "FP": [],
    }

    mix2sample = dict(N925="A", N926="D", N907="D(pH8)", N987="B", N988="C")
    sample = mix2sample[mixID]
    
    if libID == "MetaboMiner-all":
        dfMM = table1.loc[table1.groupby("Method").groups[libID]]
    elif libID == "MetaboMiner-plasma-common":
        dfMM = table1.loc[table1.groupby("Method").groups["MetaboMiner-sp"]]
        
    if sample in dfMM.groupby("Sample").groups:
        dfMM = dfMM.loc[dfMM.groupby("Sample").groups[sample]]
    else:
        # empty df
        dfMM = dfMM.iloc[0:0,:]

    df["libID"].append(libID)
    df["method"].append("MetaboMiner")
    df["ar_opt"].append(None)
    df["th_opt"].append(None)
    df["mixID"].append(mixID)
    df["F1 score"].append(dfMM["F-score"].iloc[0] / 100.0)
    df["precision"].append(dfMM["Precision"].iloc[0] / 100.0)
    df["recall"].append(dfMM["Recall"].iloc[0] / 100.0)
    df["TP"].append(dfMM["TP"].iloc[0])
    df["FN"].append(dfMM["FN"].iloc[0])
    df["FP"].append(dfMM["FP"].iloc[0])

    return pd.DataFrame(df)


def makePerformanceTableSMARTMiner(mixID):
    subs = pd.read_csv(
        SMARTMINER_DATADIR / f"predicted_{mixID}.txt", sep=";", header=None
    )
    mixSubs = loadMMMixtureLists()[mixID]

    df = {
        "libID": [],
        "F1 score": [],
        "precision": [],
        "recall": [],
        "method": [],
        "mixID": [],
        "ar_opt": [],
        "th_opt": [],
        "TP": [],
        "FN": [],
        "FP": [],
    }

    map_name2HMDB = buildName2HMDBIDMap(verb=0)
    all_names = sorted(map_name2HMDB.keys())
    detections = set()
    for name in subs.iloc[:, 0]:
        HMDB_ID = findHMDB_ID(name, map_name2HMDB, all_names)
        print(f"{name} → {HMDB_ID}")
        if HMDB_ID in detections:
            print(f"    → {HMDB_ID} ({name}) repeated.")
        detections.add(HMDB_ID)

    df["libID"].append("SMART-Miner")
    df["method"].append(f"SMART-Miner")
    df["mixID"].append(mixID)
    df["ar_opt"].append(None)
    df["th_opt"].append(None)
    TP = detections.intersection(mixSubs)
    FP = detections.difference(mixSubs)
    FN = set(mixSubs).difference(detections)
    df["TP"].append(len(TP))
    df["FP"].append(len(FP))
    df["FN"].append(len(FN))
    recall = len(TP) / (len(TP) + len(FN))
    precision = len(TP) / (len(TP) + len(FP))
    fscore = (
        precision * recall * 2 / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    df["recall"].append(recall)
    df["precision"].append(precision)
    df["F1 score"].append(fscore)

    return pd.DataFrame(df), dict(TP=TP, FP=FP, FN=FN)


def mergeOtherResults(df, colmar_match_cutoffs, libID_MM = "MetaboMiner-all"):
    classes = dict()
    df_sm_N926, classes["SMART-Miner"] = makePerformanceTableSMARTMiner("N926")
    df_sm_N907, classes["SMART-Miner"] = makePerformanceTableSMARTMiner("N907")
    df_colmar_N926_pl, classes["COLMAR"] = makePerformanceTableCOLMAR(
        "N926", match_cutoff=colmar_match_cutoffs["N926"], peaks="metabominer"
    )
    df_colmar_N926, classes["COLMAR"] = makePerformanceTableCOLMAR(
        "N926", match_cutoff=colmar_match_cutoffs["N926(dp)"]
    )
    df_colmar_N907_pl, classes["COLMAR"] = makePerformanceTableCOLMAR(
        "N907", match_cutoff=colmar_match_cutoffs["N907"], peaks="metabominer"
    )
    df_colmar_N907, classes["COLMAR"] = makePerformanceTableCOLMAR(
        "N907", match_cutoff=colmar_match_cutoffs["N907(dp)"]
    )
    df_MM = pd.concat((makePerformanceTableMM("N926", libID=libID_MM), 
                       makePerformanceTableMM("N907", libID=libID_MM)))
    df_MM.index = list(range(len(df_MM)))
    df = pd.concat(
        (
            df,
            df_sm_N907,
            df_sm_N926,
            df_colmar_N926_pl,
            df_colmar_N907_pl,
            # df_colmar_N926, # comment for manuscript figure
            # df_colmar_N907, # comment for manuscript figure
            df_MM,
        )
    )
    df.index = list(range(len(df)))
    return df, classes


def plot_spectrum_plots(
    lib, mixID, targetRasterSpec, pars, ar_fix, mixSubs, test_blank, show, zooms
):
    transfo = sqrt_transfo(250)
    plot_contained = True and not test_blank
    if plot_contained:
        # Show contained lib compounds
        contained = mixSubs[mixID[:-4]] if mixID[-4:] == "(dp)" else mixSubs[mixID]
        contained = [{c: s} for c, s in lib.items() if c in contained]
        for zoom in zooms:
            plotLibOnTarget(
                contained,
                targetRasterSpec,
                "metabominer",
                pars["targetSpecID"],
                densityTransform=transfo,
                assignmentRadius=ar_fix,
                # makeLabel=True, figsize=(6,4), # for legend plot
                makeLabel=False,
                zoom=zoom,
                figsize=(12, 10),
                outdir=outdir,
                plotTargetRaster=True,
                figtitle=targetRasterSpec.name + "_contained_cpds",
                show=show,
            )

    plot_picked_peaks = True and not test_blank
    if plot_picked_peaks:
        # Show provided peaklist
        mixPointSpec = loadMMMixPointSpectra()[mixID]
        peaklistTarget = {mixID: mixPointSpec}
        for zoom in zooms:
            plotLibOnTarget(
                peaklistTarget,
                targetRasterSpec,
                "metabominer",
                pars["targetSpecID"],
                densityTransform=transfo,
                makeLabel=False,
                figsize=(12, 10),
                outdir=outdir,
                zoom=zoom,
                figtitle=targetRasterSpec.name + "_pickedpeaks",
                plotTargetRaster=True,
                show=show,
            )

    plot_raster = True and not test_blank
    if plot_raster:
        use_sqrt_transfo = True
        if use_sqrt_transfo:
            # Plot raster spectrum
            def transfo_raster(x):
                res = np.zeros_like(x)
                try:
                    iter(x)
                    res[x > 0] = np.sqrt(x[x > 0])
                    res[x < 0] = -np.sqrt(-x[x < 0])
                except:
                    res = np.sqrt(x) if x > 0 else -np.sqrt(-x)
                return res

        else:
            transfo_raster = lambda x: x

        rc = np.max(np.abs(transfo_raster(targetRasterSpec.fullData)))
        # clim = (0, np.max(transfo_raster(targetRasterSpec.fullData)))
        clim = (-rc, rc)
        plot_ranges = {
            "y": (0, targetRasterSpec.FRanges[0][1]),
            "x": (0, targetRasterSpec.FRanges[1][1]),
        }
        plot_ranges = {"y": (50, 80), "x": (2.8, 5.2)}
        ax = plotRasterSpectrum(
            clim=clim,
            title=pars["targetSpecID"],
            densityTransform=transfo_raster,
            spec=targetRasterSpec,
            plotRange=plot_ranges,
            returnAx=True,
            colorbar=False,
            cmap="grey",
            show=show,
        )
        fig = ax.get_figure()
        fig.set_size_inches((12, 10))
        if use_sqrt_transfo:
            fn = outdir / (pars["targetSpecID"] + f"(sqrt-range {rc:g})" + ".png")
        else:
            fn = outdir / (pars["targetSpecID"] + f"(range {rc:g})" + ".png")
        fig.savefig(fn, dpi=600)
        print(f"Saved figure '{fn}'.\n")

        ax.legend(loc=2).remove()
        for zoom in zooms:
            ax.set_xlim(zoom["H"])
            ax.set_ylim(zoom["C"])
            fig.set_size_inches((5, 5))
            if use_sqrt_transfo:
                fn = outdir / (
                    pars["targetSpecID"]
                    + f"(sqrt-range {rc:g}, zoom {zoom['H']}x{zoom['C']})"
                    + ".png"
                )
            else:
                fn = outdir / (
                    pars["targetSpecID"]
                    + f"(range {rc:g}, zoom {zoom['H']}x{zoom['C']})"
                    + ".png"
                )
            fig.savefig(fn, dpi=600)
            print(f"Saved figure '{fn}'.\n")
        if show:
            plt.show()
        else:
            plt.close(fig)

    plot_noise = True
    if plot_noise:
        # Plot raster spectrum
        clim = (-targetRasterSpec.noiseStd, targetRasterSpec.noiseStd)
        plot_ranges = {
            "y": (0, targetRasterSpec.FRanges[0][1]),
            "x": (0, targetRasterSpec.FRanges[1][1]),
        }
        ax = plotRasterSpectrum(
            clim=clim,
            title=pars["targetSpecID"],
            densityTransform=transfo,
            spec=targetRasterSpec,
            plotRange=plot_ranges,
            returnAx=True,
            colorbar=False,
            show=show,
        )
        fig = ax.get_figure()
        fig.set_size_inches((12, 10))
        fn = outdir / (pars["targetSpecID"] + "_noise.png")
        fig.savefig(fn, dpi=600)
        print(f"Saved figure '{fn}'.\n")
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_fit_plots(
    targetPointSpec,
    targetRasterSpec,
    lib,
    mixSubs,
    pars,
    setup,
    mixID,
    test_blank,
    scanResults,
    assigned,
    arSpan,
    ar_fix,
    th_fix,
    show,
    zooms,
):

    def transfo(x):
        res = np.zeros_like(x)
        try:
            iter(x)
            res[x > 0] = np.minimum(np.sqrt(x[x > 0]), 250)
            res[x < 0] = np.maximum(-np.sqrt(-x[x < 0]), -250)
        except:
            res = (
                np.minimum(np.sqrt(x), 250) if x > 0 else np.maximum(-np.sqrt(-x), -250)
            )
        return res

    plot_scan = True
    if plot_scan:
        incrStr = "incr" if pars["incrementalFit"] else "onepass"
        isoStr = "iso" if pars["isolated_fit"] else "joint"
        setupTitle = f"{targetPointSpec.name} paramscan, {isoStr}-{incrStr}"
        axDict = plotPerformanceScanResults(
            scanResults,
            title=setupTitle,
            logAR=False,
            outdir=outdir,
            show=show,
            plotDetectionsInsteadOfPrecision=test_blank,
            verb=0,
            figsize=(5.0,2.0),
            plot_max_marker=False,
        )

    plot_slices = True
    if plot_slices:  # and plot:
        figname = outdir / (
            "Slices_"
            + pars["targetSpecID"]
            + f"({setup})"
            + "_th%g_ar%g_" % (th_fix, ar_fix)
            + ".svg"
        )
        figsize = (2.3, 2.7)
        # figsize = (3.0, 3.5)
        # figsize = (8, 7)
        plotSlices(
            scanResults,
            ar_fix=ar_fix,
            th_fix=th_fix,
            figname=figname,
            figsize=figsize,
            max_th=0.00055,
            report=False,
            show=show,
        )

    plot_assignment = True
    if plot_assignment:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(layout="constrained", figsize=(3, 2))
        ax.plot(arSpan * arSpan, [assigned[ar] for ar in arSpan], label="assigned")
        ax.set_xlim((min(arSpan * arSpan), max(arSpan * arSpan)))
        # ax.set_ylim((0.0, 1.0))
        ax.set_xlabel("$r^2$ (~assignment area)")
        ax.set_ylabel("total assignment")
        ax.set_title("Total assignment for " + pars["targetSpecID"])
        ax.legend()
        figname = outdir / f"ar_vs_assignment_{pars['targetSpecID']}, setup {setup}.svg"
        fig.savefig(figname)
        print(f"Saved figure '{figname}'")
        if show:
            plt.show()
        else:
            plt.close(fig)

    plot_diagnosis = True and not test_blank
    if plot_diagnosis:
        # Plot FPs, FNs, and TPs
        ID2name = buildHMDBID2NameMap(verb=0)
        scores = scanResults[ar_fix][th_fix]
        for typ in ["TP", "FP", "FN"]:
            scorelib = [{c: lib[c]} for c in scores[mixID]["classification"][typ] if c in lib]
            print(f"\n\n## {len(scores[mixID]['classification'][typ])} {typ}s of {mixID}:")
            for cd in scorelib:
                c, s = list(cd.items())[0]
                pp(
                    {
                        c
                        + " ("
                        + ID2name[c]
                        + ")": dict(weights=s.weights, coords=s.coords)
                    }
                )
                # # Debug
                # from shapely.geometry import Point
                # target = mixSpecs[mixID]
                # region = target.regions[0]
                # for pt in s.coords:
                #     print(f"Point {pt} in target region: %s"%(region.contains(Point(pt))))

            for zoom in zooms:
                plotLibOnTarget(
                    scorelib,
                    targetRasterSpec,
                    "metabominer",
                    pars["targetSpecID"],
                    assignmentRadius=ar_fix,
                    markersize=1,
                    densityTransform=transfo,
                    makeLabel=True,
                    figsize=(12, 10),
                    outdir=outdir,
                    radius_alpha=0.2,
                    zoom=zoom,
                    figtitle=targetRasterSpec.name
                    + f"_{setup}_"
                    + typ
                    + f"_ar{ar_fix:g}_th{th_fix:g}",
                    show=show,
                    plotTargetRaster=True,
                    # plot legends
                    # plotTargetRaster=False,
                    # ext="svg",
                )
        # Plot FPs vs contained:
        FPmix = makeMix(
            {c: lib[c] for c in scores[mixID]["classification"]["FP"] if c in lib}, name="FPs"
        )
        FNmix = makeMix(
            {c: lib[c] for c in scores[mixID]["classification"]["FN"] if c in lib},
            name="FNs",
        )
        TPmix = makeMix(
            {c: lib[c] for c in scores[mixID]["classification"]["TP"] if c in lib}, name="TPs"
        )
        # contained_mix = makeMix({c: lib[c] for c in mixSubs[mixID] if c in lib}, name="contained")

        for zoom in zooms:
            plotLibOnTarget(
                [{"FPs": FPmix}, {"FNs": FNmix}, {"TPs": TPmix}],
                targetRasterSpec,
                "metabominer",
                pars["targetSpecID"],
                assignmentRadius=ar_fix,
                markersize=1,
                densityTransform=transfo,
                makeLabel=True,
                figsize=(12, 10),
                outdir=outdir,
                radius_alpha=0.2,
                zoom=zoom,
                figtitle=targetRasterSpec.name
                + f"_{setup}_FPvsFNvsTP_ar{ar_fix:g}_th{th_fix:g}",
                show=show,
                plotTargetRaster=True,
                # plot legends
                # plotTargetRaster=False,
                # ext="svg",
            )


def report_performance(df, classes):
    # Report FN and FP stats
    fp_counts, fn_counts = defaultdict(list), defaultdict(list)
    for method in classes.keys():
        print(method)
        print(sorted(classes[method]["FP"]))
        for cpd in classes[method]["FP"]:
            fp_counts[cpd].append(method)
        for cpd in classes[method]["FN"]:
            fn_counts[cpd].append(method)

    ID2name = buildHMDBID2NameMap(verb=0)
    cpds = sorted(fp_counts.keys())
    counts = [len(fp_counts[cpd]) for cpd in cpds]
    ixs = np.argsort([f"{cnt}-{cpd}" for cnt,cpd in zip(counts, cpds)])
    print("\n# False Positive counts:")
    for ix in ixs:
        print(
            f"{ID2name[cpds[ix]]} (HMDB-ID: {cpds[ix]}): {counts[ix]} ({fp_counts[cpds[ix]]})"
        )
    cpds = sorted(fn_counts.keys())
    counts = [len(fn_counts[cpd]) for cpd in cpds]
    ixs = np.argsort([f"{cnt}-{cpd}" for cnt,cpd in zip(counts, cpds)])
    print("\n# False Negative counts:")
    for ix in ixs:
        print(
            f"{ID2name[cpds[ix]]} (HMDB-ID: {cpds[ix]}): {counts[ix]} ({fn_counts[cpds[ix]]})"
        )
    
    print("\nOptimal F1 scores")
    print(df.columns)
    methods = sorted(set(df["method"]))
    g = df.groupby("method").groups
    for m in methods:
        print("#", m)
        ix = g[m]
        dfg = df.loc[ix,:]
        mixIDs = sorted(set(dfg["mixID"]))
        gg = dfg.groupby("mixID").groups
        for mixID in mixIDs:
            ixx = gg[mixID]
            fscore = list(dfg.loc[ixx, "F1 score"])[0]
            th = list(dfg.loc[ixx, "th_opt"])[0]
            ar = list(dfg.loc[ixx, "ar_opt"])[0]
            print(f"{mixID}: F1 = {fscore:g} (ar={ar:g}, th={th:g})")
        print()
    
    


def run_sample_diagnosis(
    mixID="N926",
    ar_fix=0.3,
    th_fix=0.0,
    test_blank=False,
    blank_seed=None,
    plot_spectrum=True,
    plot_fit=True,
    setup=None,
    show=False,
    zooms=[],
    load=False,
    libName="metabominer",
):

    if setup == "D(dp)":
        # Use setup D on the deeppicker peaklist
        setup = "D"
        deeppicker = True
    else:
        deeppicker = False

    noiseFilterLevel = None
    smoothing = 0.0
    # smoothing = 0.5

    lib = loadMMLibrary(libName=libName)
    mixSubs = loadMMMixtureLists()

    arSpan = [
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
    ]
    if mixID in ["N907", "N926"]:
        thSpan = np.linspace(0.0, 0.001, 41)
    else:
        # This has other scales
        thSpan = np.linspace(0.0, 0.015, 31)

    arSpan = np.array([np.round(ar, 7) for ar in arSpan])
    thSpan = np.array([np.round(th, 7) for th in thSpan])

    print("Check lib containment...")
    for ID, contained in zip(
        sorted(mixSubs[mixID]), np.isin(sorted(mixSubs[mixID]), list(lib))
    ):
        if not contained:
            print(f"HMDB-ID {ID} is not in MM-library.")
            if ID == "294":
                print("(HMDB-ID 294 denotes Urea)\n")

    setupSpec = case_dict[setup][mixID]
    bins = binning[mixID]
    kwargs = dict(
        assignmentRadius=None,
        incrementalFit=setupSpec["incrementalFit"],
        isolated_fit=setupSpec["isolated_fit"],
        pointTarget=setupSpec["pointTarget"],
        smoothing=smoothing,
        noiseFilterLevel=noiseFilterLevel,
        binning=bins,
        workingDir=cache_dir,
    )
    pars = prepareParameters(
        specDomain="metabominer", targetSpecID=mixID, libID=libName, **kwargs
    )

    # denoising_pars = dict(smoothRadius=1.0, noiseFilterLevel=None)
    denoising_pars = dict(smoothRadius=smoothing, noiseFilterLevel=noiseFilterLevel)
    targetRasterSpec = loadMetaboMinerSpectrum(mixID, denoising_pars=denoising_pars)

    if test_blank:
        seed = blank_seed
        blankRasterSpec = sampleBlankSpectrum(targetRasterSpec, seed)
        targetRasterSpec = blankRasterSpec
        pars["targetSpecID"] += "(blank, seed%d)" % seed

    if pars["pointTarget"]:
        if deeppicker:
            mixID += "(dp)"
            targetPointSpec = loadMMMixPointSpectra()[mixID]
        else:
            targetPointSpec = loadMMMixPointSpectra()[mixID]
        targetPointSpec.regions = []
    else:
        targetPointSpec = makeRasterPointSpectrum(
            targetRasterSpec, nbin=bins, signalThreshold=noiseFilterLevel
        )

    # Diagnostic plots prior to fit
    if plot_spectrum:
        plot_spectrum_plots(
            lib, mixID, targetRasterSpec, pars, ar_fix, mixSubs, test_blank, show, zooms
        )

    ## Run scan
    data = dict()
    assigned = {}

    if pars["pointTarget"]:
        mixSpecs = loadMMMixPointSpectra()
    else:
        pars["binning"] = binning[mixID]
        mixSpecs = collect_and_serialize_MM_grid_spectra([mixID], pars)

    if pars["incrementalFit"]:
        pars["arSpan"] = arSpan
        results = runMetaboMinerFits(
            base_pars=pars,
            lib=lib,
            mixSpecs=mixSpecs,
            mixIDs=[mixID],
            returnAllResults=True,
            load=load,
        )
        for ar in arSpan:
            pars["assignmentRadius"] = ar
            dfTargets, dfCompounds = makeResultsDF(
                [{"res": results[mixID][ar], "pars": pars}],
                originalWeights=totalWeights(lib),
            )
            data[ar] = {
                mixID: {
                    "targets": dfTargets,
                    "compounds": dfCompounds,
                    "targetSpecID": pars["targetSpecID"],
                    "targetTotalWeight": results[mixID][ar].originalWeightY,
                }
            }
    else:
        for ar in arSpan:
            pars["assignmentRadius"] = ar
            results = runMetaboMinerFits(
                # base_pars=pars, lib=lib, mixSpecs=mixSpecs, mixIDs=[mixID], load=False
                base_pars=pars,
                lib=lib,
                mixSpecs=mixSpecs,
                mixIDs=[mixID],
                load=load,
            )
            result = results[mixID]

            dfTargets, dfCompounds = makeResultsDF(
                [{"res": result, "pars": pars}], originalWeights=totalWeights(lib)
            )
            data[ar] = {
                mixID: {
                    "targets": dfTargets,
                    "compounds": dfCompounds,
                    "targetSpecID": pars["targetSpecID"],
                    "targetTotalWeight": result.originalWeightY,
                }
            }

    scanResults = defaultdict(dict)
    for ar in arSpan:
        assigned[ar] = data[ar][mixID]["targets"]["assigned"].iloc[0]
        assigned[ar] *= data[ar][mixID]["targets"]["originalWeightY"].iloc[0]
        for th in thSpan:
            scores = classificationStats(
                data[ar], pars, th, mixSubs, [mixID], scale_by_peaknr=True, verb=False
            )
            scanResults[ar][th] = scores

    if plot_fit:
        plot_fit_plots(
            targetPointSpec,
            targetRasterSpec,
            lib,
            mixSubs,
            pars,
            setup,
            mixID,
            test_blank,
            scanResults,
            assigned,
            arSpan,
            ar_fix,
            th_fix,
            show,
            zooms,
        )

    return scanResults, assigned


def selectParameters(scanResults, assigned):
    targetIDs = sorted(scanResults)
    mixIDs = sorted(set(targetID.split("_")[0] for targetID in targetIDs))
    arSpan = np.array(sorted(scanResults[targetIDs[0]]))

    slope_range = [0.12, 0.15]
    for mixID in mixIDs:
        blankID = mixID + "_blank"
        ar1, ar2 = slope_range
        # slope of ar^2 vs assined
        slope_blank = (assigned[blankID][ar2] - assigned[blankID][ar1]) / (
            ar2 * ar2 - ar1 * ar1
        )
        print(f"\n{mixID}:")
        print("   blank assignment slope over (%g, %g): %g" % (ar1, ar2, slope_blank))
        for ar1, ar2 in zip(arSpan[:-1], arSpan[1:]):
            slope_blank = (assigned[blankID][ar2] - assigned[blankID][ar1]) / (
                ar2 * ar2 - ar1 * ar1
            )
            print("       (%g, %g): %g" % (ar1, ar2, slope_blank))
        print("     mix assignment slopes:")
        for ar1, ar2 in zip(arSpan[:-1], arSpan[1:]):
            slope_mix = (assigned[mixID][ar2] - assigned[mixID][ar1]) / (
                ar2 * ar2 - ar1 * ar1
            )
            print("       (%g, %g): %g" % (ar1, ar2, slope_mix))


def buildComparisonDataframe(scanResults):
    df = {
        "libID": [],
        "F1 score": [],
        "precision": [],
        "recall": [],
        "method": [],
        "mixID": [],
        "ar_opt": [],
        "th_opt": [],
        "TP": [],
        "FN": [],
        "FP": [],
    }
    classes = defaultdict(dict)

    for setup in scanResults:
        for mixID, data in scanResults[setup].items():
            ar_opt, th_opt = None, None
            maxf1 = -np.inf
            tp, fp, fn = 0, 0, 0
            for ar in data:
                for th, scores in data[ar].items():
                    if setup[-4:] == "(dp)":
                        s = scores[mixID + "(dp)"]["classification"]
                    else:
                        s = scores[mixID]["classification"]
                    f1 = s["fscore"]
                    if f1 > maxf1:
                        maxf1 = f1
                        ar_opt, th_opt = ar, th
                        tp, fp, fn = len(s["TP"]), len(s["FP"]), len(s["FN"])
                        TPs, FPs, FNs = s["TP"], s["FP"], s["FN"]
                        recall, precision = s["recall"], s["precision"]
            df["libID"].append(metaboMinerLibNames[0])
            df["F1 score"].append(maxf1)
            df["precision"].append(precision)
            df["recall"].append(recall)
            df["TP"].append(tp)
            df["FP"].append(fp)
            df["FN"].append(fn)
            df["ar_opt"].append(ar_opt)
            df["th_opt"].append(th_opt)
            df["method"].append(setup)
            df["mixID"].append(mixID)
        classes[setup]["TP"] = TPs
        classes[setup]["FP"] = FPs
        classes[setup]["FN"] = FNs
    return pd.DataFrame(df), classes


def run(save, mixIDs, setups, ar_fix, th_fix, load, plot, show, libName=None):
    
    if libName is None:
        libName = "metabominer"

    # Zoom display for raster plots
    zooms = [{"C": [53, 80], "H": [2.8, 5.2]}, {"C": [30, 53], "H": [2.2, 3.8]}]
    scanResults, assigned = defaultdict(dict), defaultdict(dict)
    for mixID in mixIDs:
        for setup in setups:
            # # Debug: blank spectrum analysis
            # scanResults[mixID + "_blank"], assigned[mixID + "_blank"] = run_sample_diagnosis(
            #     mixID=mixID, ar_fix=ar_fix, th_fix=th_fix,
            #     test_blank=True, blank_seed=123,
            # plot_spectrum=False, plot_fit=False,
            # )
            scanResults[f"MCF-{setup}"][mixID], assigned[f"MCF-{setup}"][mixID] = (
                run_sample_diagnosis(
                    mixID=mixID,
                    ar_fix=ar_fix,
                    th_fix=th_fix,
                    setup=setup,
                    plot_spectrum=False,
                    plot_fit=plot,
                    zooms=zooms,
                    load=load,
                    show=show,
                    libName=libName
                )
            )

    df, classes = buildComparisonDataframe(scanResults)

    if save:
        tmpfn = get_mcfnmr_home() / "tmp" / f"MCFNMR_comp_df_lib_{libName}.pickle"
        with open(tmpfn, "wb") as f:
            pickle.dump((df, classes), f)
        print(f"Saved '{tmpfn}'")
    else:
        return df, classes

    # # Debug: Deprecated approach trying to detect optimal radius and threshold
    # # by looking at detections in blank spectrum. Sample specific noise
    # # characteristics are hard to emulate, though.
    # selectParameters(scanResults, assigned)


def main(load=False, plot=False, libName="MetaboMiner-all"):
    mixIDs = ["N926", "N907"]
    # mixIDs = ["N926"]
    setups = ["A", "B", "C", "D", "D(dp)"]
    # setups = ["A", "B", "C", "D"] # For manuscript figure
    # setups = ["C"] # For manuscript figure
    ar_fix = 0.05
    th_fix = 0.000125
    
    # Debug
    # save==True generates file tmp/comp_df.pickle,
    # used to avoid lengthy collection of scan results for MCF
    save = True
    show = False
    res = run(
        save,
        mixIDs=mixIDs,
        setups=setups,
        ar_fix=ar_fix,
        th_fix=th_fix,
        load=load,
        plot=plot,
        show=show,
        libName=libName,
    )

    if not save:
        df, classes_mcf = res
    else:
        # Expects saved result from run(True)
        tmpfn = get_mcfnmr_home() / "tmp" / f"MCFNMR_comp_df_lib_{libName}.pickle"
        with open(tmpfn, "rb") as f:
            df, classes_mcf = pickle.load(f)
            print(f"Loaded '{tmpfn}'")
    
    colmar_match_cutoffs = {"N926(dp)": 0.9, "N907(dp)": 0.9,
                            "N926": 0.8, "N907": 0.8}
    libID_MM = "MetaboMiner-all" if libName.lower() == "metabominer" else libName
    df, classes = mergeOtherResults(df, colmar_match_cutoffs=colmar_match_cutoffs,
                                    libID_MM = libID_MM)
    classes.update(classes_mcf)

    import seaborn as sns


    plot_comparison = True
    if plot_comparison:
        pal = sns.color_palette("magma", n_colors=7)
        pal = pal[:3] + [pal[3]] * 2 + pal[4:6] + pal[5:] # comment for manuscript figure
        plotMetaboMinerComparison(
            None,
            df,
            plotOnlyFScore=True,
            title=f"Comparison plasma sample (lib {libName})",
            show=show,
            show_legend=False,
            # show_legend=True, # comment for manuscript figure
            plot_width=3.7,
            plot_height=2.3,
            pal=pal,
        )

    report = True
    if report:
        report_performance(df, classes)


if __name__ == "__main__":
    # libName = "metabominer" # → "MetaboMiner-all" (version for paper)
    # libName = "MetaboMiner-common"
    # libName = "MetaboMiner-plasma-all"
    # libName = "MetaboMiner-plasma-common"

    # main(load=True, plot=False, libName="metabominer")
    main(load=True, plot=False, libName="MetaboMiner-plasma-common")
