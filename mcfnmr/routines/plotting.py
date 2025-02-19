import os
import numpy as np
from pprint import pp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import starmap

from mcfnmr.demodata import (
    mixNamesOLDB,
    expectedConcentrationFactors,
    compoundLongNames,
    correctShortCompoundLabels,
    scanNumbersMixes, textbook_abbreviations,
)
from mcfnmr.demodata.loading import loadOLDBCompoundLib, buildHMDBID2NameMap
from mcfnmr.routines.utils import (
    totalWeights,
    figs_dir_MM,
    figs_dir_oldb,
    filterDF,
    figs_dir,
)
from mcfnmr.config import DEBUG, OUTDIR


def plotPredictionDiagonals(dfCompounds_dict, ar, setup=None, show=False):
    if setup:
        isoFit, incrFit = setup
    dfs = list()
    for k, d in dfCompounds_dict.items():
        if setup and setup != k:
            continue
        for mixID, df in d.items():
            g = df.groupby("assignment radius")
            df = df.loc[g.groups[ar]]
            if setup:
                g = df.groupby("incremental fit")
                df = df.loc[g.groups[incrFit]]
                g = df.groupby("isolated fit")
                df = df.loc[g.groups[isoFit]]
            dfs.append(df)
    df = pd.concat(dfs)

    make_setup_str = lambda iso, incr: ("iso-" if iso else "joint-") + (
        "incr" if incr else "onepass"
    )
    m = list(starmap(make_setup_str, zip(df["isolated fit"], df["incremental fit"])))
    df["setup"] = list(m)
    df["predicted"] = df["concentration factor"] * 3
    df["expected"] = np.round(df["expected concentration factor"] * 3, 10)
    df["signal-to-noise"] = df["expected"] * np.sqrt(
        [scanNumbersMixes[targetID] for targetID in df["target"]]
    )
    
    if setup:
        df.to_csv(figs_dir / f"quantitative_predictions_compounds_setup{setup}.csv", sep=",")
    else:
        df.to_csv(figs_dir / "quantitative_predictions_compounds.csv", sep=",")
    
    # Jitter out expected in Y-axis to get
    # a better view on point density
    jfac = 0.06
    rng = np.random.default_rng(123)
    jitter = 1 + 2 * jfac * rng.random(len(df["expected"])) - jfac
    df["expected (jittered)"] = df["expected"] * jitter

    fig, ax = plt.subplots(layout="constrained", figsize=(4.0, 4.0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([0.01, 50])
    ax.set_xlim([0.1, 40])
    if setup:
        sns.stripplot(df, y="predicted", x="expected", native_scale=True, s=3, ax=ax, zorder=0)
        sns.boxplot(df, y="predicted", x="expected", native_scale=True, 
                    fliersize=0, boxprops=dict(alpha=0.75), ax=ax, zorder=1)
    else:
        sns.stripplot(df, y="predicted", x="expected", hue="setup", native_scale=True, s=3, ax=ax, zorder=0)
        sns.boxplot(df, y="predicted", x="expected", hue="setup", native_scale=True, 
                    fliersize=0, boxprops=dict(alpha=0.75), ax=ax, zorder=1)
    ax.plot([0, 50], [0, 50], "k--", lw=0.5, zorder=-2)
    # ax.set_ylim([0.0, 50])
    # ax.set_xlim([0.0, 40])
    ax.set_title("predicted vs expected concentrations at ar=%g (excl. exp.=0)" % ar)

    if setup:
        setupstr = df["setup"].iloc[0]
        figname = (
            figs_dir / f"predicted_vs_expected_concentrations_hue_by_S2N_{setupstr}.svg"
        )
    else:
        figname = figs_dir / "predicted_vs_expected_concentrations_hue_by_setup.svg"
    fig.savefig(figname)
    print(f"Saved figure as '{figname}'")
    
    plot_single_compounds = True
    if plot_single_compounds:
        R2s = {}
        loglog = True
        ext = "svg"
        df.index = list(range(len(df)))
        if setup is None or setup == (False, True):
            if loglog:
                figsubdir = figs_dir / "concentration_prediction_per_cpd_loglog"
            else:
                figsubdir = figs_dir / "concentration_prediction_per_cpd"
            if not figsubdir.exists():
                os.makedirs(figsubdir)
            
            gg = df.groupby("compound").groups
            for c in sorted(set(df["compound"])):
                ix = gg[c]
                if setup:
                    figname = figsubdir / f"expected_predicted_{c}_setup{setup}.{ext}"
                else:
                    figname = figsubdir / f"expected_predicted_{c}.{ext}"
                dfc = df.iloc[ix,:]
                fig, ax = plt.subplots(figsize = (2.0,2.0), layout="constrained")
                sns.scatterplot(dfc, y="predicted", x="expected", hue="setup", s=10, ax=ax, zorder=0)
                if loglog:
                    ax.set_yscale("log")
                    ax.set_xscale("log")
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                lims = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
                lims = (0.01, lims[1]) # for indentical xlims for manuscript figures
                ax.set_xlim(lims), ax.set_ylim(lims)
                ax.plot(lims,lims, "k--", lw=0.5, zorder=-2)
                
                # Fit linear regression and plot regression curve
                import statsmodels.api as sm
                if loglog:
                    mod = sm.OLS(dfc["predicted"], dfc["expected"])
                else:
                    # Fit with intercept
                    dfc["Intercept"] = 1.0
                    mod = sm.OLS(dfc["predicted"], dfc[["Intercept", "expected"]])
                res = mod.fit()
                print(res.summary())
                a = 0.0 if loglog else res.params["Intercept"]
                b = res.params["expected"]
                R2 = res.rsquared
                R2s[compoundLongNames[c]]=R2
                avg_err = np.nanmean(np.abs(dfc["predicted"] - dfc["expected"])/dfc["expected"])
                # xspan = np.exp(np.linspace(*np.log(lims), 101))
                xspan = np.linspace(*lims, 101)
                ax.plot(xspan, a+b*xspan, color="red", zorder=-1)
                ax.set_title(textbook_abbreviations[compoundLongNames[c]] + f" ($R^2={R2:g}$, avg.err={avg_err:g})")
                fig.savefig(figname)
                print(f"Saved figure '{figname}'")
                # plt.show()
                plt.close(fig)
            print("\nR2S:")
            pp(R2s)
            
    if show:
        plt.show()


def plotAssignmentStatsOLDB(
    dfCompounds,
    dfTargets,
    libID,
    targetID,
    showCompounds=None,
    markCompounds=[],
    outdir=None,
    show=False,
    setupStr="",
):
    assert outdir is not None
    targetName = mixNamesOLDB.get(targetID, targetID)
    if markCompounds is None:
        markCompounds = []
    arSpan = sorted(set(dfCompounds["assignment radius"]))

    # gathers info during preparing plots (return of this function)
    stats = {}

    errorStats = {
        "mean relative error": [],
        "percentage mid range": [],
        "std(err)": [],
        "assignment radius": [],
        "mixture": [],
        "matching": [],
        "false positives": [],
    }

    # For each assignment radius, plot the single compound stats
    nrows = len(arSpan)

    # Setup main figure: selected compound's match vs expectation
    # and error distributions for different ar
    plt.style.use("bmh")
    fig_main = plt.figure(figsize=(3.3, 2.3))  # , layout="tight")
    fig, fig2 = fig_main.subfigures(nrows=1, ncols=2, width_ratios=(1, 1))
    # fig.get_layout_engine().set(hspace=10, wspace=10)
    for i in range(nrows):
        ax1 = fig.add_subplot(nrows, 1, i + 1)
        ax2 = fig2.add_subplot(nrows, 1, i + 1)
        if i != nrows - 1:
            ax1.set_xlabel("")
            ax2.set_xlabel("")
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])

    axes, axes2 = fig.get_axes(), fig2.get_axes()
    fig.suptitle("Assignment (%s)" % (setupStr))
    fig2.suptitle("Prediction error (%s)" % (setupStr))

    # Extra plot for the four compounds with higher concetrations
    selectedCompounds = ["Bnz3", "Gluc", "Pim", "Tyr"]
    nExpect = len(selectedCompounds)

    if nrows == 0:
        raise Exception("plotAssignmentStatsOLDB(): error")

    fig1, axes1 = plt.subplots(
        ncols=nrows, figsize=(2 * nrows, min(10, nExpect * 0.75))
    )
    fig1.suptitle("Assignment present (%s)" % (setupStr))

    filters = {"lib": libID, "target": targetName}
    dfC = filterDF(dfCompounds, filters)
    dfT = filterDF(dfTargets, filters)
    totalWeightY = dfT["originalWeightY"].iloc[0]
    lib = loadOLDBCompoundLib()
    cweights = totalWeights(lib)

    print(f"\n\n#### {setupStr} ####")
    # Ensure axes are iterable
    if nrows == 1:
        axes1 = [axes1]

    for ar, ax, ax1, ax2 in zip(arSpan, axes, axes1, axes2):
        dfr = dfC.loc[dfC["assignment radius"] == ar].copy()
        dfr.loc[:, "expected"] = [
            expectedConcentrationFactors[targetID][compoundLongNames.get(c, c)]
            for c in dfr["compound"]
        ]
        totalDetected = sum(
            dfr["concentration factor"]
            * np.array([cweights[compoundLongNames.get(c, c)] for c in dfr["compound"]])
        )
        totalExpected = sum(
            dfr["expected"]
            * np.array([cweights[compoundLongNames.get(c, c)] for c in dfr["compound"]])
        )

        print(f"\nAssignment Radius {ar}:")
        print(f"    totalWeightY = {totalWeightY}")
        print(f"    totalExpected = {totalExpected}")
        print(f"    totalDetected = {totalDetected}")

        for c, cf, ex in zip(
            dfr["compound"], dfr["concentration factor"], dfr["expected"]
        ):
            if ex > 0:
                print(f"{c}: cf/ex = %g" % (cf / ex))

        dfr.loc[:, "error"] = (
            dfr.loc[:, "concentration factor"] - dfr.loc[:, "expected"]
        )
        dfr.loc[:, "rel. error"] = dfr.loc[:, "error"] / dfr.loc[:, "expected"]
        infrows = dfr["rel. error"] == np.inf
        dfr.loc[infrows, "rel. error"] = np.nan
        uniform_matchtype = (
            len(set(dfr["isolated fit"])) == 1 and len(set(dfr["incremental fit"])) == 1
        )
        if uniform_matchtype:
            isoFitStr = "indep." if list(set(dfr["isolated fit"]))[0] else "joint"
            incrFitStr = (
                " incr. fit" if list(set(dfr["incremental fit"]))[0] else " fit"
            )
            match_type = f"{isoFitStr}{incrFitStr}"
        else:
            match_type = "various"

        # Calculate mean relative error and midrange percentage
        # for present compounds
        dfr_present = dfr[dfr["expected"] > 0]
        std = np.nanstd(dfr_present["rel. error"])
        meanErr = np.nanmean(dfr_present["rel. error"])
        meanAbsErr = np.nanmean(np.abs(dfr_present["rel. error"]))
        pctInMidrange = (
            100
            * np.count_nonzero(
                np.logical_and(
                    dfr_present["rel. error"] > -0.5, dfr_present["rel. error"] < 1.0
                )
            )
            / len(dfr_present["rel. error"])
        )

        # Check if any compounds which are not present have benn detected
        fp_thresh = np.max(dfr["expected"]) * 0.01
        FP = dfr["compound"][
            (dfr["expected"] == 0) & (dfr["concentration factor"] > fp_thresh)
        ]
        if len(FP) > 0:
            print(f"Found false positive detection for {targetID}: {FP}")
        errorStats["false positives"].append(len(FP))
        errorStats["mean relative error"].append(meanAbsErr)
        errorStats["percentage mid range"].append(pctInMidrange)
        errorStats["std(err)"].append(std)
        errorStats["assignment radius"].append(ar)
        errorStats["mixture"].append(targetID)
        errorStats["matching"].append(match_type)

        dfr.loc[:, "in mix"] = np.isin(dfr.loc[:, "compound"], markCompounds)
        dfr_expected = dfr.loc[np.isin(dfr.loc[:, "compound"], selectedCompounds)]
        if len(dfr["assigned"]) == 0:
            continue

        # Filter data to only show thespecified set of compounds
        if showCompounds is not None:
            ix = [c in showCompounds for c in dfr["compound"]]
            dfrshow = dfr[ix]
        else:
            dfrshow = dfr

        shortlabs = correctShortCompoundLabels(dfrshow["compound"])
        dfrshow.loc[:, "compound"] = shortlabs
        shortlabs = correctShortCompoundLabels(dfr_expected["compound"])
        dfr_expected.loc[:, "compound"] = shortlabs
        dodge = True

        sns.barplot(
            dfrshow,
            x="compound",
            y="concentration factor",
            color="xkcd:grass",
            dodge=dodge,
            ax=ax,
        )
        sns.barplot(
            dfrshow,
            x="compound",
            y="expected",
            color="xkcd:yellow",
            dodge=dodge,
            alpha=0.5,
            ax=ax,
        )

        if len(markCompounds) > 0:
            for label in ax.get_yticklabels():
                col = "xkcd:green" if label.get_text() in markCompounds else "xkcd:red"
                label.set_color(col)
        ylim = (0.0, 1.5 * max(dfr["expected"]))
        ax.set_ylim(ylim)
        ax.text(0.2, ylim[1] * 0.9, "$r = %g$" % ar, va="top")
        if ax == axes[-1]:
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
            ax.set_ylabel("")
        if ax == axes[0]:
            ax.set_ylabel("")

        ax1.set_title("$r = %g$" % ar)
        sns.barplot(
            dfr_expected,
            x="compound",
            y="concentration factor",
            color="xkcd:grass",
            dodge=dodge,
            ax=ax1,
        )
        sns.barplot(
            dfr_expected,
            x="compound",
            y="expected",
            color="xkcd:yellow",
            dodge=dodge,
            alpha=0.5,
            ax=ax1,
        )
        ax1.set_ylim((0.0, 3 * max(dfr["expected"])))

        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)

        pal = sns.color_palette("viridis", n_colors=5)
        histRange = [-1.1, 2.5]
        nbins = 18

        if len(set(dfr["expected"])) > 1:
            ylims = [0, 25]
            ho = (max(dfr["expected"]), min(dfr["expected"]))
            sns.histplot(
                dfr,
                x="rel. error",
                hue="expected",
                hue_order=ho,
                multiple="layer",
                stat="percent",
                common_norm=False,
                bins=nbins,
                binrange=histRange,
                ax=ax2,
                palette=[pal[0], pal[3]],
            )
        else:
            ylims = [0, 12]
            sns.histplot(
                dfr,
                x="rel. error",
                stat="count",
                bins=nbins,
                binrange=histRange,
                color=pal[0],
                ax=ax2,
            )
        ax2.set_ylim(ylims)
        ax2.plot([meanErr, meanErr], ylims, ls="--", lw=1.0, color=pal[0], label="mean")
        ylims = ax2.get_ylim()
        ax2.set_ylim(ylims)
        ax2.plot([-0.5, -0.5], ylims, ls="--", lw=1.0, color="xkcd:salmon")
        ax2.plot([1, 1], ylims, ls="--", lw=1.0, color="xkcd:salmon")
        ax2title = "$r = %g$, std=%g, %s in [1/2, 2]: %.1f" % (
            ar,
            std,
            "%",
            pctInMidrange,
        )
        ax2.set_title(ax2title)

    stats["errors"] = errorStats

    figname = outdir / f"assignment and prediction_err {setupStr}.png"
    fig_main.savefig(figname)
    figname = outdir / f"assignment and prediction_err {setupStr}.svg"
    fig_main.savefig(figname)
    print(f"Saved fig {figname}")
    if not show:
        plt.close(fig_main)

    fig1.tight_layout()
    figname = outdir / f"compoundAssignment to expected {setupStr}.png"
    fig1.savefig(figname)
    figname = outdir / f"compoundAssignment to expected {setupStr}.svg"
    fig1.savefig(figname)
    print(f"Saved fig {figname}")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    return stats


def plotErrorStats(data, figname, show=True):
    """
    For different experimental series, plot assignment radius against
    mean rel. error and percentage mid range
    """

    # Experimental series
    series = {
        # all cpds:
        "all": ["Ia_01", "Ib_01", "Ic_01"],
        # 4 cpds:
        "4 cpds": ["IIa_01", "IIb_01", "IIc_01"],
        # all + 4 cpds:
        "all+4": ["IIIa_01", "IIIb_01", "IIIc_01"],
    }

    df = pd.DataFrame(data)

    mix_descriptions = {
        "Ia_01": r"$N_{\mathrm{mix}}=34$,\n$c_k^\ast=3\ mM$,",
        "Ib_01": r"$N_{\mathrm{mix}}=34$,\n$c_k^\ast=0.3\ mM$,",
        "Ic_01": r"$N_{\mathrm{mix}}=34$,\n$c_k^\ast=0.03\ mM$,",
        "IIa_01": r"$N_{\mathrm{mix}}=4$,\n$c_k^\ast=30\ mM$,",
        "IIb_01": r"$N_{\mathrm{mix}}=4$,\n$c_k^\ast=3\ mM$,",
        "IIc_01": r"$N_{\mathrm{mix}}=4$,\n$c_k^\ast=0.3\ mM$,",
        "IIIa_01": r"$N_{\mathrm{mix}}=34$,\n$c_k^\ast=\{0.15, 15.15\}\ mM$,",
        "IIIb_01": r"$N_{\mathrm{mix}}=34$,\n$c_k^\ast=\{0.15, 1.65\}\ mM$,",
        "IIIc_01": r"$N_{\mathrm{mix}}=34$,\n$c_k^\ast=\{0.15, 0.30\}\ mM$,",
    }
    for k in mix_descriptions.keys():
        nScans = scanNumbersMixes[k]
        mix_descriptions[k] += "\n" + r"$\mathrm{NS}=" + ("%d$" % nScans)

    ylims = {
        "all": dict(midrange=[0, 105], relerr=[0, 2.2]),
        "4 cpds": dict(midrange=[0, 105], relerr=[0, 2.2], fps=[0, 34]),
        "all+4": dict(midrange=[0, 105], relerr=[0, 2.2]),
    }

    plt.style.use("bmh")
    fig = plt.figure(figsize=(10, 12))
    figs = fig.subfigures(3, 1, wspace=0.01)

    for s, f in zip(series, figs):
        four_cpds = False
        mixes = series[s]
        f.suptitle(s + f" ({mixes})")
        # reference axes for sharing y-axis
        for i, m in enumerate(mixes):
            dfm = df[df["mixture"] == m]

            print(f"Mix {m}:")
            for h in sorted(set(dfm["matching"])):
                dfh = dfm[dfm["matching"] == h]
                minErrIx = np.argmin(dfh["mean relative error"])
                minErrAr = dfh["assignment radius"].iloc[minErrIx]
                minErr = dfh["mean relative error"].iloc[minErrIx]
                maxMidrngIx = np.argmax(dfh["percentage mid range"])
                maxMidrngAr = dfh["assignment radius"].iloc[maxMidrngIx]
                maxMidrng = dfh["percentage mid range"].iloc[maxMidrngIx]
                if DEBUG > 0:
                    print(f"  setup {h}:")
                    print("    min rel err: %g (at r=%g)" % (minErr, minErrAr))
                    print("    max mid rng: %g (at r=%g)" % (maxMidrng, maxMidrngAr))

            ax1 = f.add_subplot(
                2 + four_cpds, len(mixes), i + 1
            )  # , sharey=share_ax_err)
            sns.lineplot(
                dfm,
                x="assignment radius",
                y="mean relative error",
                hue="matching",
                ax=ax1,
            )
            if m in ["0203_01", "0203_01", "0207_01"]:
                ax1.text(0.21, 1.05, mix_descriptions[m], va="top")
            else:
                ax1.text(0.01, 2.05, mix_descriptions[m], va="top")
            ax1.legend(loc=2).remove()
            ax1.set_ylim(ylims[s]["relerr"])
            ax1.set_xlim((0, 0.5))

            ax2 = f.add_subplot(
                2 + four_cpds, len(mixes), len(mixes) + i + 1
            )  # , sharey=share_ax_mid)
            sns.lineplot(
                dfm,
                x="assignment radius",
                y="percentage mid range",
                hue="matching",
                ax=ax2,
            )
            ax2.set_ylim(ylims[s]["midrange"])
            ax2.set_xlim((0, 0.5))

            if four_cpds:
                ax3 = f.add_subplot(
                    2 + four_cpds, len(mixes), 2 * len(mixes) + i + 1
                )  # , sharey=share_ax_mid)
                sns.lineplot(
                    dfm,
                    x="assignment radius",
                    y="false positives",
                    hue="matching",
                    ax=ax3,
                )
                ax3.set_ylim(ylims[s]["fps"])
                ax3.legend(loc=2).remove()
                ax3.set_xlim((0, 0.5))

            # Remove unnecessary labels
            ax1.set_xlabel("")
            ax1.set_xticklabels("")
            if i != 0:
                ax1.set_ylabel("")
                ax2.set_ylabel("")
                ax1.set_yticklabels("")
                ax2.set_yticklabels("")
            if i != len(mixes) - 1 or s != "all":
                ax2.legend(loc=2).remove()

    fn = figs_dir_oldb / f"{figname}.png"
    fig.savefig(fn)
    fn = figs_dir_oldb / f"{figname}.svg"
    fig.savefig(fn)
    print(f"Saved fig {fn}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plotTimings(results, setup, mixIDs, loglog=True, show=True):
    fig, ax = plt.subplots(layout="constrained", figsize=(4, 3))
    # fig, ax = plt.subplots(layout="constrained", figsize=(6,4))
    mcfTimes = {}
    colors = ["xkcd:blue", "xkcd:orange", "xkcd:green"]
    for i, mixID in enumerate(mixIDs):
        timings = {ar: res[mixID].timings for ar, res in results.items()}
        arSpan = np.array(sorted(timings))
        if mixID == mixIDs[0]:
            ax.plot(arSpan * arSpan, np.zeros_like(arSpan), "k--", lw=1)
        if setup == "setup D":
            mcfTimesSecs = [timings[ar]["minCostFlow"] for ar in arSpan]
            mcfTimes[mixID] = mcfTimesSecs
        else:
            mcfTimesMins = [timings[ar]["minCostFlow"] / 60 for ar in arSpan]
            mcfTimes[mixID] = mcfTimesMins
        if loglog:
            ax.plot(arSpan, mcfTimes[mixID], marker=".", label=mixID, color=colors[i])
        else:
            ax.plot(
                arSpan * arSpan,
                mcfTimes[mixID],
                marker=".",
                label=mixID,
                color=colors[i],
            )
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("assignment radius $r$")
        ax.set_xlim(arSpan[[0, -1]])
    else:
        ticks = arSpan * arSpan
        labels = ["$%g\\times%g$" % (ar, ar) for ar in arSpan]
        ax.set_xticks(ticks, labels, rotation=45)
        ax.set_xlabel("assignment area $r^2$")
        ax.set_xlim((arSpan * arSpan)[[0, -1]])
    ax.set_title(f"Timings linprog() {setup}")
    if setup == "setup D":
        ax.set_ylabel("time [secs]")
    else:
        ax.set_ylabel("time [mins]")
    ax.legend()

    print("timings:")
    pp(mcfTimes)

    # Test asymptotic, effective complexity
    asr_ix = [-5, -1]  # asymptotic range
    print("asymptotic range:", arSpan[asr_ix])
    logt = {mixID: np.log(mcfTimes[mixID])[asr_ix] for mixID in mixIDs}
    log_asr = np.log(arSpan[asr_ix])
    slopes = {
        mixID: (logt[mixID][1] - logt[mixID][0]) / (log_asr[1] - log_asr[0])
        for mixID in mixIDs
    }
    print("log-log slopes (time~ar^alpha):")
    pp(slopes)

    if loglog:
        for i, mixID in enumerate(mixIDs):
            ax.plot(
                arSpan[asr_ix],
                np.array(mcfTimes[mixID])[asr_ix],
                ls="--",
                color=colors[i],
            )

    logstr = "_loglog" if loglog else ""
    figname = figs_dir / f"Timings_{setup}{logstr}.svg"
    fig.savefig(figname)
    print(f"Saved fig '{figname}'")

    if show:
        plt.show()


def plotPerformanceScanResults(
    scanResults,
    title=None,
    logAR=True,
    plotPcolor=True,
    plotContour=False,
    outdir=None,
    show=False,
    plotDetectionsInsteadOfPrecision=False,
    verb=1,
    figsize=None,
    plot_max_marker=True,
):

    arSpan = sorted(scanResults.keys())
    thSpan = sorted(scanResults[arSpan[0]].keys())
    nar, nth = len(arSpan), len(thSpan)

    # Collect data into matrices
    fscores = np.zeros((nar, nth))
    recalls = np.zeros((nar, nth))
    precisions = np.zeros((nar, nth))
    detections = np.zeros((nar, nth))
    max_ix, maxfscore = (0, 0), 0.0
    for i, ar in enumerate(arSpan):
        if verb > 0:
            print("\nar=%g" % ar)
        for j, th in enumerate(thSpan):
            if verb > 0:
                print("th=%g" % th)
            stats = scanResults[ar][th]
            mixIDs = sorted(stats.keys())
            data = [stats[mixID]["classification"] for mixID in mixIDs]
            fscores[i, j] = np.mean(np.array([d["fscore"] for d in data]))
            recalls[i, j] = np.mean(np.array([d["recall"] for d in data]))
            precisions[i, j] = np.mean(np.array([d["precision"] for d in data]))
            detections[i, j] = np.sum(
                np.array([len(d["TP"]) + len(d["FP"]) for d in data])
            )
            if verb > 0:
                print("fscore=%g" % fscores[i, j])
                print("recall=%g" % recalls[i, j])
                print("precision=%g" % precisions[i, j])

            if fscores[i, j] > maxfscore:
                max_ix = (i, j)
                maxfscore = fscores[i, j]
    armax, thmax = arSpan[max_ix[0]], thSpan[max_ix[1]]
    print("\n### Maximal fscore: %g (at: ar=%g, th=%g)" % (maxfscore, armax, thmax))
    print(
        "    recall: %g, precision: %g"
        % (recalls[max_ix[0], max_ix[1]], precisions[max_ix[0], max_ix[1]])
    )

    maxstats = dict(
        [
            (mixID, scanResults[armax][thmax][mixID]["classification"])
            for mixID in mixIDs
        ]
    )
    for mixID, data in maxstats.items():
        print("  mix %s:" % mixID)
        print("      recall: %g" % data["recall"])
        print("      precision: %g" % data["precision"])
        print("      fscore: %g" % data["fscore"])

    # Plot contour
    X, Y = np.meshgrid(thSpan, arSpan)
    if figsize is None:
        # figsize = (5,2.) # For main text 2x2
        figsize = (8, 2.7)  # For supplement 1x
    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=figsize)
    fig.suptitle(title)
    axes[0].set_ylabel("assignment radius")
    if plotDetectionsInsteadOfPrecision:
        labs = ["F1 score", "recall", "detections"]
        fields = [fscores, recalls, detections]
    else:
        labs = ["F1 score", "recall", "precision"]
        fields = [fscores, recalls, precisions]
    for ax, vals, lab in zip(axes, fields, labs):
        if plotPcolor:
            if lab == "detections":
                cf = ax.pcolormesh(X, Y, vals, cmap="magma")
            else:
                cf = ax.pcolormesh(X, Y, vals, vmin=0.0, vmax=1.0, cmap="magma")
            if plotContour:
                ax.contour(
                    X,
                    Y,
                    vals,
                    vmin=0.0,
                    vmax=1.0,
                    levels=np.linspace(0.0, 1.0, 11),
                    zorder=100,
                    linewidths=0.5,
                    alpha=0.5,
                    cmap="Greys",
                )
        else:
            cf = ax.contourf(
                X,
                Y,
                vals,
                vmin=0.0,
                vmax=1.0,
                cmap="inferno",
                levels=np.linspace(0.0, 1.0, 11),
            )
        
        if plot_max_marker:
            ax.scatter(thmax, armax, s=15, c="xkcd:red", marker="+", linewidths=0.8)
        ax.set_title(lab)
        ax.set_xlabel("detection threshold")
        # ax.grid(color="#eeeeee", zorder=1)
        if logAR:
            ax.set_yscale("log")
    # fig.colorbar(cf)
    plt.tight_layout()

    if outdir is None:
        outdir = figs_dir_MM
    figname = outdir / (title + ".png")
    fig.savefig(figname, dpi=600)
    print(f"Saved fig '{figname}'")
    figname = outdir / (title + ".svg")
    fig.savefig(figname, dpi=600)
    print(f"Saved fig '{figname}'")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plotSlices(
    scanResults, ar_fix, th_fix, figname, figsize, report=True, max_th=None, show=True
):
    scores = scanResults
    mixID = sorted(scanResults[ar_fix][th_fix].keys())[0]
    plt.style.use("bmh")
    arSpan = sorted(scanResults.keys())
    thSpan = sorted(scanResults[arSpan[0]].keys())

    fig, axes = plt.subplots(nrows=2, layout="constrained", figsize=figsize)
    data = {th: scanResults[ar_fix][th][mixID]["classification"] for th in thSpan}
    fscores = [data[th]["fscore"] for th in thSpan]
    # recall    = [data[th]["recall"] for th in thSpan]
    # precision = [data[th]["precision"] for th in thSpan]
    TPs = [len(data[th]["TP"]) for th in thSpan]
    FPs = [len(data[th]["FP"]) for th in thSpan]
    FNs = [len(data[th]["FN"]) for th in thSpan]
    ax2 = axes[0]
    ax2.plot(thSpan, TPs, label="confirmed")
    ax2.plot(thSpan, FNs, label="undetected")
    ax2.plot(thSpan, FPs, label="unconfirmed")
    ax2.set_title("Slice at $r=%g$" % ar_fix)
    ax2.set_xlabel("detection threshold")
    ax2.set_ylabel("compounds")
    ax2.set_ylim((-0.5, 50))
    if max_th:
        ax2.set_xlim((thSpan[0], max_th))
    else:
        ax2.set_xlim((thSpan[0], thSpan[-1]))

    print("\nMaximal 'fscore' within th-slice at ar=%g: %g" % (ar_fix, max(fscores)))

    data = {ar: scanResults[ar][th_fix][mixID]["classification"] for ar in arSpan}
    fscores = [data[ar]["fscore"] for ar in arSpan]
    # recall    = [data[ar]["recall"] for ar in arSpan]
    # precision = [data[ar]["precision"] for ar in arSpan]
    TPs = [len(data[ar]["TP"]) for ar in arSpan]
    FPs = [len(data[ar]["FP"]) for ar in arSpan]
    FNs = [len(data[ar]["FN"]) for ar in arSpan]
    ax2 = axes[1]
    ax2.plot(arSpan, TPs)
    ax2.plot(arSpan, FNs)
    ax2.plot(arSpan, FPs)
    fig.legend()
    ax2.set_title("Slice at $\\vartheta=%g$" % th_fix)
    ax2.set_xlabel("assignment radius")
    ax2.set_ylabel("compounds")
    ax2.set_ylim((-0.5, 50))
    ax2.set_xlim((arSpan[0], arSpan[-1]))

    print("Maximal 'fscore' within ar-slice at th=%g: %g\n" % (th_fix, max(fscores)))

    if figname:
        fig.savefig(figname)
        print(f"Saved figure '{figname}'")

    if report:
        id2name = buildHMDBID2NameMap(verb=0)
        print("\n# plotSlices() detection report:")
        prev = {}
        for th in reversed(thSpan):
            data = scanResults[ar_fix][th][mixID]["classification"]
            print(f"\nAt ar={ar_fix:g}, th={th:g}:")
            print(f"   # TP: {len(data['TP'])}")
            print(f"   # FP: {len(data['FP'])}")
            print(f"   # FN: {len(data['FN'])}")
            newTPs = set(data["TP"]).difference(prev.get("TP", []))
            newFPs = set(data["FP"]).difference(prev.get("FP", []))
            print(f"new TPs:")
            for ID in newTPs:
                print(f"  {ID} ({id2name[ID]}),")
            print(f"new FPs:")
            for ID in newFPs:
                print(f"  {ID} ({id2name[ID]}),")
            prev = data

    if show:
        plt.show()
    else:
        plt.close(fig)
