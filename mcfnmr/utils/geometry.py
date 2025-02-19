import pickle
import os
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.affinity import scale
from mcfnmr.utils import parsesvg


class RectRegion(object):
    def __init__(self, xrange, yrange):
        self.xrange = xrange
        self.yrange = yrange

    def hitTest(self, xy):
        x, y = xy
        return (
            x >= self.xrange[0]
            and x <= self.xrange[1]
            and y >= self.yrange[0]
            and y <= self.yrange[1]
        )


def load_regions(regions):
    if regions:
        if type(regions) is dict:
            res = parsesvg.loadRegions(**regions)
        else:
            # regions already provided as Polygon list
            assert type(regions) is list
            res = regions
    else:
        res = []
    return res


def getIndicesWithin(regions, coords, target_id=None, cache_dir=None, load=False):
    """
    Return coord indices, which are not part of any given region
    """
    if target_id is not None:
        if cache_dir is None:
            raise Exception("Need working directory to use index-lookup-cache.")
        fn = cache_dir / (target_id + "_region_ix.pickle")
        if load and os.path.exists(fn):
            with open(fn, "rb") as f:
                ix = pickle.load(f)
            print("Loaded indices from %s" % fn)
            return ix
    else:
        fn = None

    if len(regions) == 0:
        return np.ones(coords.shape[0], dtype=bool)
    regionUnion = regions[0]
    for i in range(1, len(regions)):
        regionUnion = regionUnion.union(regions[i])
    ix = np.array([regionUnion.contains(Point(xy)) for xy in coords])

    if fn is not None:
        with open(fn, "wb") as f:
            pickle.dump(ix, f)
        print("Saved indices to %s" % fn)

    return ix


def makeRadiusPoly(xy, rx, ry, typ="ellipse"):
    # Make a square
    x, y = xy
    if typ == "rect":
        # draw rect
        path = ((x - rx, y - ry), (x + rx, y - ry), (x + rx, y + ry), (x - rx, y + ry))
        poly = Polygon(shell=path)
    else:
        # draw ellipse
        circ = Point(x, y).buffer(1)
        poly = scale(circ, rx, ry)  # type(ellipse)=polygon

    return poly
