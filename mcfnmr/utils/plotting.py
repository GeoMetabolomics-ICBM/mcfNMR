import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import seaborn as sns

from mcfnmr.config import DEFAULT_SCALE_H, DEFAULT_SCALE_C
from mcfnmr.demodata import (
    textbook_abbreviations,
)
from mcfnmr.utils.pointspectrum import (
    makeMix,
    detect_grid,
)
from mcfnmr.routines.utils import figs_dir
from mcfnmr.utils.parsesvg import swapAxes
from mcfnmr.utils.geometry import makeRadiusPoly
from argparse import Namespace
from mcfnmr.demodata.loading import buildHMDBID2NameMap


def sqrt_transfo(max_x):
    def transfo(x):
        res = np.zeros_like(x)
        try:
            iter(x)
            res[x > 0] = np.minimum(np.sqrt(x[x > 0]), max_x)
            res[x < 0] = np.maximum(-np.sqrt(-x[x < 0]), -max_x)
        except:
            res = (
                np.minimum(np.sqrt(x), max_x)
                if x > 0
                else np.maximum(-np.sqrt(-x), -max_x)
            )
        return res

    return transfo


def plotLibOnTarget(
    lib,
    targetSpec,
    libID,
    targetSpecID="",
    assignmentRadius=0.1,
    plotTargetRaster=True,
    densityTransform=None,
    compound_numbers=False,
    plotRadii=True,
    m_alpha=1.0,
    radius_alpha=0.2,
    markerAlphaFromIntensity=True,
    markersize=10,
    scaleSizes=True,
    zoom=None,
    ext="png",
    figsize=None,
    makeLabel=False,
    outdir=figs_dir,
    figtitle=None,
    plotTitle=None,
    show=False,
    ax=None,
):
    if type(lib) is list:
        # Giving a list of libs is used for assigning different colors to peaks (each lib gets a proper one)
        allMixed = []
        for l in lib:
            allMixed.append(makeMix(basis=l, alpha=dict([(k, 1.0) for k in l.keys()])))
            if len(l) == 1:
                allMixed[-1].name = list(l.values())[0].name
    else:
        allMixed = makeMix(basis=lib, alpha=dict([(k, 1.0) for k in lib.keys()]))
    plotPointMixOnRasterTarget(
        allMixed,
        targetSpec,
        targetSpecID,
        libID,
        assignmentRadius=assignmentRadius,
        densityTransform=densityTransform,
        nMix=1,
        outdir=outdir,
        plotTargetRaster=plotTargetRaster,
        flipAxes=True,
        plotRadii=plotRadii,
        m_alpha=m_alpha,
        radius_alpha=radius_alpha,
        makeLabel=makeLabel,
        markersize=markersize,
        compound_numbers=compound_numbers,
        markerAlphaFromIntensity=markerAlphaFromIntensity,
        scaleSizes=scaleSizes,
        plotRegions=False,
        zoom=zoom,
        figsize=figsize,
        ext=ext,
        figtitle=figtitle,
        plotTitle=plotTitle,
        show=show,
        ax=ax,
    )


def plotPointMixOnRasterTarget(
    mix,
    targetSpec,
    targetSpecID,
    libID,
    assignmentRadius,
    nMix,
    outdir,
    flipAxes=True,
    invertAxes=True,
    plotRadii=True,
    plotRegions=True,
    plotTargetRaster=True,
    densityTransform=None,
    figsize=None,
    figsize_zoom=None,
    makeLabel=False,
    markerAlphaFromIntensity=True,
    markersize=10,
    scaleSizes=True,
    compound_numbers=False,
    zoom=None,
    ext="png",
    m_alpha=1.0,
    radius_alpha=0.2,
    show=False,
    figtitle=None,
    plotTitle=None,
    ax=None,
):
    # Don't show if ax was given
    ax_given = ax is not None
    show = show and not ax_given

    if plotTitle is None and plotTargetRaster:
        plotTitle = "targetSpec %s approx. by %s (nMix %s, ar %g)" % (
            targetSpecID,
            libID,
            nMix,
            assignmentRadius,
        )
    elif plotTitle is None:
        plotTitle = "Lib '%s' mix peaks approx %s (nMix %s, ar %g)" % (
            libID,
            targetSpecID,
            nMix,
            assignmentRadius,
        )
    if figtitle is None:
        figfn = outdir / (plotTitle + "." + ext)
    else:
        figfn = outdir / (figtitle + "." + ext)

    flipAxesImPlot = not flipAxes  # imageplot uses different axis ordering

    mpl_style = "default"
    plt.style.use(mpl_style)

    if plotTargetRaster:
        # Raster background plot
        clim = (0, np.max(targetSpec.fullData))
        plot_ranges = {"y": targetSpec.FRanges[0], "x": targetSpec.FRanges[1]}
        ax = plotRasterSpectrum(
            clim=clim,
            title=plotTitle,
            densityTransform=densityTransform,
            spec=targetSpec,
            plotRange=plot_ranges,
            flipAxis=flipAxesImPlot,
            colorbar=False,
            returnAx=True,
            ax=ax,
        )
    else:
        if ax is None:
            _, ax = plt.subplots()
        ax.set_title(plotTitle)
        ax.set_xlabel("")
        if flipAxes:
            ax.set_ylabel("C-shift [ppm]")
            ax.set_xlabel("H-shift [ppm]")
        else:
            ax.set_ylabel("H-shift [ppm]")
            ax.set_xlabel("C-shift [ppm]")

    if plotRegions:
        ax = plotRegionPolys(targetSpec.regions, z0=100, givenAx=ax, flipAxes=flipAxes)

    if plotRadii:
        radii = (assignmentRadius * DEFAULT_SCALE_C, assignmentRadius * DEFAULT_SCALE_H)
    else:
        radii = None

    if type(mix) is list:
        # mix is a list of libs that should be plotted in differen colors
        if targetSpecID == "Ia_01":
            # hack: re-order to control color of specific compounds for plotting OLDB-lib
            #       (this is the only place where I used this approach so far)
            # TODO: rather solve this on caller side!
            # swaps: (21 ,2), (12, 25)
            mix[2], mix[21] = mix[21], mix[2]
            mix[25], mix[12] = mix[12], mix[25]
            print(f"Re-ordering compounds for {targetSpecID}!")

        for i, m in enumerate(mix):
            lab = m.name if makeLabel else None
            lab = textbook_abbreviations.get(lab, lab)
            if makeLabel and compound_numbers:
                lab = ("%d: " % i) + lab

            weights = m.weights if scaleSizes else None
            ax = plotPeaks(
                m.coords,
                weights=weights,
                pcol_ix=i,
                mcol_ix=i,
                radii=radii,
                m_alpha=m_alpha,
                markersize=markersize,
                radius_alpha=radius_alpha,
                iterate_marker_shape=len(mix) > 9,
                alphaFromIntensity=markerAlphaFromIntensity,
                z0=300,
                givenAx=ax,
                flipAxes=flipAxes,
                label=lab,
            )
        if makeLabel:
            # ax.legend(loc=2, ncols=2)
            ax.legend(loc=2, ncols=4)
    else:
        weights = mix.weights if scaleSizes else None
        ax = plotPeaks(
            mix.coords,
            weights=weights,
            radii=radii,
            z0=300,
            m_alpha=m_alpha,
            markersize=markersize,
            radius_alpha=radius_alpha,
            alphaFromIntensity=markerAlphaFromIntensity,
            givenAx=ax,
            flipAxes=flipAxes,
            label=targetSpecID,
        )

    if flipAxes:
        ax.set_xlim(targetSpec.FRanges[1])
        ax.set_ylim(targetSpec.FRanges[0])
    else:
        ax.set_xlim(targetSpec.FRanges[0])
        ax.set_ylim(targetSpec.FRanges[1])
    if invertAxes:
        ax.invert_xaxis()
        ax.invert_yaxis()

    fig = ax.get_figure()
    if figsize_zoom is None:
        figsize = (5, 5)
    fig.set_size_inches(figsize)
    if not ax_given:
        # Caller takes responsibility
        fig.savefig(figfn, dpi=600)
        print("Saved figure '%s'" % figfn)
    if zoom is not None:
        if flipAxes:
            ax.set_xlim(zoom["H"])
            ax.set_ylim(zoom["C"])
        else:
            ax.set_xlim(zoom["C"])
            ax.set_ylim(zoom["H"])
        if invertAxes:
            ax.invert_xaxis()
            ax.invert_yaxis()
        if figtitle is None:
            figzoomfn = outdir / (
                plotTitle + "_zoom%sx%s." % (zoom["H"], zoom["C"]) + ext
            )
        else:
            figzoomfn = outdir / (
                figtitle + "_zoom%sx%s." % (zoom["H"], zoom["C"]) + ext
            )
        ax.legend(loc=2).remove()
        if not ax_given:
            fig.savefig(figzoomfn, dpi=600)
    if show:
        plt.show()
    elif not ax_given:
        plt.close(fig)


def plotRasterSpectrum(
    spec,
    title,
    clim=None,
    plotRange=None,
    flipAxis=False,
    invertAxes=True,
    densityTransform=None,
    colorbar=True,
    cmap="whitejet",
    show=False,
    returnAx=True,
    ax=None,
):
    if cmap == "whitejet":
        # Colormap for spectral intensity
        colors = [
            "white",
            "white",
            "xkcd:red",
            "xkcd:yellow",
            "xkcd:turquoise",
            "xkcd:deep blue",
        ]
        nodes = [0.0, 0.01, 0.15, 0.3, 0.7, 1.0]
        cmap = LinearSegmentedColormap.from_list(
            "intensity_cmap", list(zip(nodes, colors))
        )
    elif cmap in ["grey", "gray"]:
        # Diverging blue->white->red to be transferable back to intensities from image
        colors = [
            "#ffffff",
            "#000000",
        ]
        nodes = [0.0, 1.0]
        cmap = LinearSegmentedColormap.from_list(
            "intensity_cmap", list(zip(nodes, colors))
        )
    elif cmap == "div":
        # Diverging blue->white->red to be transferable back to intensities from image
        colors = [
            "#0000ff",
            "#ffffff",
            "#ff0000",
        ]
        nodes = [0.0, 0.5, 1.0]
        cmap = LinearSegmentedColormap.from_list(
            "intensity_cmap", list(zip(nodes, colors))
        )
    else:
        raise Exception(f"Unknown cmap '{cmap}'")

    if flipAxis:
        data, ranges = spec.fullData.T, (spec.FRanges[1], spec.FRanges[0])
        if plotRange is not None:
            plotRange = {"y": plotRange["x"], "x": plotRange["y"]}
    else:
        data, ranges = spec.fullData, spec.FRanges

    if clim is None:
        clim = (np.min(data), np.max(data))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    dims = data.shape
    xspan = np.linspace(ranges[0][0], ranges[0][1], dims[0])
    yspan = np.linspace(ranges[1][0], ranges[1][1], dims[1])

    if densityTransform is not None:
        print("max(data) = %g" % np.nanmax(data))
        print("min(data) = %g" % np.nanmin(data))
        data = densityTransform(data)
        print("max(Transform(data)) = %g" % np.nanmax(data))
        print("min(Transform(data)) = %g" % np.nanmin(data))
        clim = (densityTransform(clim[0]), densityTransform(clim[1]))
        m = ax.pcolormesh(yspan, xspan, data, vmin=clim[0], vmax=clim[1], cmap=cmap)
    else:
        print("max(data) = %g" % np.max(data))
        print("min(data) = %g" % np.min(data))
        m = ax.pcolormesh(yspan, xspan, data, vmin=clim[0], vmax=clim[1], cmap=cmap)

    if flipAxis:
        ax.set_ylabel("H-shift [ppm]")
        ax.set_xlabel("C-shift [ppm]")
    else:
        ax.set_ylabel("C-shift [ppm]")
        ax.set_xlabel("H-shift [ppm]")

    if colorbar:
        fig.colorbar(m, ax=ax)

    if plotRange:
        ax.set_xlim(plotRange["x"])
        ax.set_ylim(plotRange["y"])
    ax.set_title(title)

    if invertAxes:
        ax.invert_xaxis()
        ax.invert_yaxis()

    if show:
        plt.show()
    else:
        if returnAx:
            return ax
        else:
            plt.close(fig)


def plotRegionPolys(regions, z0=0, givenAx=None, show=True, flipAxes=False):
    if givenAx is None:
        _, ax = plt.subplots()
    else:
        ax = givenAx

    if regions is None:
        regions = []

    if flipAxes:
        regions = swapAxes(regions)

    for z, poly in enumerate(regions):
        plot_polygon(
            ax, poly, zorder=z0 + z, facecolor="#66666699", edgecolor="#0000FFFF"
        )

    if givenAx is not None:
        return ax
    elif show:
        plt.show()


# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
    )

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plotPeaks(
    coords,
    weights=None,
    pcol_ix=-1,
    mcol_ix=-1,
    markPtIx=None,
    label=None,
    radii=None,
    z0=0,
    m_alpha=1.0,
    alphaFromIntensity=False,
    radius_alpha=0.2,
    markersize=10,
    iterate_marker_shape=True,
    givenAx=None,
    flipAxes=False,
):
    peakColors = sns.color_palette("bright", 9)
    markerColors = peakColors
    markerShapes = [".", "v", "^", "*", "+", "x", "D"]
    if pcol_ix == -1 or mcol_ix == -1:
        pcol = mcol = "black"
        pshp = "."
    else:
        pcol = peakColors[pcol_ix % len(peakColors)]
        mcol = markerColors[mcol_ix % len(markerColors)]
        if iterate_marker_shape:
            pshp = markerShapes[mcol_ix % len(markerShapes)]
        else:
            pshp = markerShapes[0]

    if givenAx is None:
        _, ax = plt.subplots()
    else:
        ax = givenAx

    if weights is None:
        weights = np.ones(len(coords))
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    # srange for weight in [0,1]
    srange = (markersize, markersize)
    # srange = (0.5, 0.5) # For small fig zoom
    sizes = srange[0] + (srange[1] - srange[0]) * weights

    if len(coords) > 0:
        x, y = coords.T
        if radii is not None:
            for xi, yi in zip(x, y):
                if flipAxes:
                    radius = makeRadiusPoly((yi, xi), rx=radii[1], ry=radii[0])
                else:
                    radius = makeRadiusPoly((xi, yi), rx=radii[0], ry=radii[1])
                plot_polygon(ax, radius, color=pcol, alpha=radius_alpha, zorder=z0 + 1)

        if markPtIx is not None:
            if flipAxes:
                ax.scatter(
                    x,
                    y,
                    color="gray",
                    s=sizes,
                    marker=pshp,
                    zorder=z0 + 1,
                    linewidths=0.0,
                )
            else:
                ax.scatter(
                    y,
                    x,
                    color="gray",
                    s=sizes,
                    marker=pshp,
                    zorder=z0 + 1,
                    linewidths=0.0,
                )
            x, y = coords[markPtIx, :].T
            sizes = sizes[markPtIx]
        if alphaFromIntensity:
            alpha = m_alpha + weights * (1.0 - m_alpha) / np.max(weights)
        else:
            alpha = m_alpha
        if flipAxes:
            ax.scatter(
                y,
                x,
                color=mcol,
                s=sizes,
                marker=pshp,
                alpha=alpha,
                zorder=z0 + 3,
                label=label,
            )
        else:
            ax.scatter(
                x,
                y,
                color=mcol,
                s=sizes,
                marker=pshp,
                alpha=alpha,
                zorder=z0 + 3,
                label=label,
            )

    return ax


def plot_detected(df, lib, target, assignment_radius, figname, libname, show=False):
    # Plot point spectrum with detected compounds
    # This is the plotting function called from user interface mcfNMR
    figsize = (10, 8)
    title = ".".join(figname.name.split(".")[:-1])
    show_legend = True

    # Check if the target is generated from a grid spectrum
    # If so, plot target as heatmap.
    res = detect_grid(target.coords)
    grid_spec = res[0] is not None

    predicted = sorted(c for c, d in zip(df["compound"], df["detection"]) if d)

    if grid_spec:
        xspan, yspan, indexer = res
        Z = indexer.makeMatrix(target.weights)
        grid_spec_mock = Namespace(
            FRanges=((min(xspan), max(xspan)), (min(yspan), max(yspan))),
            fullData=Z,
        )
        q95 = np.quantile(target.weights, 0.95)
        ax = plotRasterSpectrum(
            grid_spec_mock,
            title=title,
            colorbar=True,
            returnAx=True,
            densityTransform=sqrt_transfo(q95 * 2),
        )
        fig = ax.get_figure()
    else:
        print(f"Could not deduce grid for spectrum.\n   ({res[1]})")
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        FRanges = (
            (min(target.coords[:, 0]), max(target.coords[:, 0])),
            (min(target.coords[:, 1]), max(target.coords[:, 1])),
        )
        grid_spec_mock = Namespace(FRanges=FRanges)

        # Plot target point spectrum
        target_peaks = {target.name: target}
        plotLibOnTarget(
            target_peaks,
            grid_spec_mock,
            libID=libname,
            targetSpecID=target.name,
            assignmentRadius=0.02,
            markersize=0.1,
            radius_alpha=0.2,
            plotTargetRaster=False,
            makeLabel=show_legend,
            ax=ax,
        )

    if libname == "MetaboMiner - Biofluid ( all )":
        try:
            ID2name = buildHMDBID2NameMap(verb=0)
            predictions = [
                {ID2name[str(c)]: spec} for c, spec in lib.items() if c in predicted
            ]
        except:
            predictions = [{c: spec} for c, spec in lib.items() if c in predicted]
    else:
        predictions = [{c: spec} for c, spec in lib.items() if c in predicted]

    title = ".".join(figname.name.split(".")[:-1])
    plotLibOnTarget(
        predictions,
        grid_spec_mock,
        libID=libname,
        targetSpecID=target.name,
        assignmentRadius=assignment_radius,
        plotTargetRaster=False,
        figsize=figsize,
        makeLabel=show_legend,
        plotTitle=title,
        ax=ax,
    )

    fig.set_size_inches(figsize)
    fig.savefig(figname)
    print(f"Saved figure '{figname}'")

    if show:
        plt.show()
    else:
        plt.close(fig)
