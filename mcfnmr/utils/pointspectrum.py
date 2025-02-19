import numpy as np
from pprint import pformat
import scipy

from mcfnmr.utils.geometry import RectRegion
from itertools import product


class PointSpectrum(object):
    def __init__(
        self,
        coords,
        weights,
        spec=None,
        fromRaster=False,
        binning=None,
        pointArea=None,
        name=None,
        regions=[],
    ):
        assert not (spec is None and fromRaster)
        assert not (binning is None and fromRaster)
        assert not (pointArea is None and fromRaster)
        if len(weights) != len(coords):
            raise Exception(f"Length of weights != length of coords for '{name}'")
        if np.any(np.isnan(weights)):
            raise Exception(f"NaN in weight array for '{name}'")
        self.coords = np.array(coords)
        self.coords.shape = (self.coords.shape[0], 2)
        if len(weights) != len(coords):
            raise Exception(
                f"Length of weights != length of coords for '{name}' after reshape..."
            )
        self.weights = np.array(weights).flatten()
        self.unknownRegions = []
        self.representsRasterData = fromRaster
        self.binning = binning
        self.pointDimensions = None
        self.pointArea = pointArea
        self.name = name
        self.regions = regions

        if spec is not None:
            self.scanNr = spec.scanNr
            self.noiseStd = spec.noiseStd
            self.regions = spec.regions
            self.pointDimensions = (
                int(spec.pointDimensions[0] / binning[0]),
                int(spec.pointDimensions[1] / binning[1]),
            )
            self.FRanges = spec.FRanges

        if self.coords.size == 0:
            self.coords.shape = (0, 2)
        assert self.coords.size == 2 * self.weights.size

    def isPointSpectrum(self):
        return True

    def size(self):
        return len(self.weights)

    def __str__(self):
        s = "Pointspectrum (size: %d)\n" % self.size()
        s += "  weights:\n"
        s += pformat(self.weights, indent=3)
        s += "\n  coords:\n"
        s += pformat(self.coords, indent=3)
        return s

    def mult(self, x):
        return PointSpectrum(self.coords, x * self.weights)

    def add(self, spec):
        return PointSpectrum(
            np.concatenate((self.coords, spec.coords)),
            np.concatenate((self.weights, spec.weights)),
        )

    def isEmpty(self):
        return self.weights.size == 0


def emptyPointSpectrum():
    return PointSpectrum([], [])


def makeMix(basis, alpha=None, name=None):
    if alpha is None:
        alpha = {c: 1.0 for c in basis}
    # Create mixture
    X = emptyPointSpectrum()
    for k in alpha.keys():
        if k not in basis:
            print("Key '%s' not in basis. Ignoring..." % k)
            continue
        b = basis[k]
        if np.any(np.isnan(b.weights)):
            raise Exception("Malformed intensity for compound %s!" % k)
        X = X.add(b.mult(alpha[k]))
    if name is not None:
        X.name = name
    return X


def makeRasterPointSpectrum(rasterspec, nbin=(1, 1), signalThreshold=None, xVar="F1"):
    coords, weights, pointClusterArea = rasterspec.getRasterDataAsPeaks(
        nbin=nbin, signalThreshold=signalThreshold, xVar=xVar
    )
    spec = PointSpectrum(
        coords,
        weights,
        spec=rasterspec,
        fromRaster=True,
        binning=nbin,
        pointArea=pointClusterArea,
        name=rasterspec.name,
    )
    return spec


def constructRasterFromPointSpectrum(spec, rasterDim, rasterRange, kernelWidths):
    range0 = rasterRange[0][1] - rasterRange[0][0]
    range1 = rasterRange[1][1] - rasterRange[1][0]
    dy, dx = range0 / rasterDim[0], range1 / rasterDim[1]

    # Construct kernel to be added to spectrum around coords in Z
    # We assume that the kernel width corresponds to 4 stds of
    # the corresponding normal distribution. For the kernel data, we
    # keep 6 stds
    kernelDims = [
        int(np.ceil(kernelWidths[1] * 3 / dy)),
        int(np.ceil(kernelWidths[0] * 3 / dx)),
    ]
    # Assume have odd dimension for centering
    if kernelDims[0] % 2 == 0:
        kernelDims[0] += 1
    if kernelDims[1] % 2 == 0:
        kernelDims[1] += 1
    # Central point index in kernel
    ix0 = ((kernelDims[0] - 1) / 2, (kernelDims[1] - 1) / 2)

    kernelRelY = (np.arange(kernelDims[0]) - ix0[0]) * dx
    kernelRelX = (np.arange(kernelDims[1]) - ix0[1]) * dy
    kernelRelX, kernelRelY = np.meshgrid(kernelRelX, kernelRelY)
    vary = kernelWidths[0] * kernelWidths[0] / 16
    varx = kernelWidths[1] * kernelWidths[1] / 16
    d = scipy.stats.multivariate_normal(mean=[0.0, 0.0], cov=[[vary, 0], [0, varx]])
    kernelRelXY = np.array(list(zip(kernelRelY.flatten(), kernelRelX.flatten())))
    kernel = d.pdf(kernelRelXY)
    kernel.shape = kernelRelX.shape
    # Make shape round
    ks = kernel.shape
    for i0 in range(ks[0]):
        for i1 in range(ks[1]):
            if (
                np.linalg.norm((2 * (i0 - ix0[0]) / ks[0], 2 * (i1 - ix0[1]) / ks[1]))
                > 1
            ):
                kernel[i0, i1] = 0

    # Debug
    # pcm = plt.pcolormesh(kernelRelY, kernelRelX, kernel, cmap="hot")
    # plt.show()

    # make grid of pixel centerpoints
    yspan = np.linspace(
        rasterRange[0][0] + dy / 2, rasterRange[0][1] - dy / 2, rasterDim[0]
    )
    xspan = np.linspace(
        rasterRange[1][0] + dx / 2, rasterRange[1][1] - dx / 2, rasterDim[1]
    )
    X, Y = np.meshgrid(xspan, yspan)
    Z = np.zeros_like(X)

    # Add kernels concentrated at single peaks
    ixrad = int((kernelDims[0] - 1) / 2), int((kernelDims[1] - 1) / 2)
    region = RectRegion(xrange=rasterRange[0], yrange=rasterRange[1])
    for xy, w in zip(spec.coords, spec.weights):
        if not region.hitTest(xy):
            # Ignore peaks outside of target range
            continue
        # Find closest coord index for xy
        yix = int(np.round((rasterDim[0] - 1) * (xy[0] - rasterRange[0][0]) / range0))
        xix = int(np.round((rasterDim[1] - 1) * (xy[1] - rasterRange[1][0]) / range1))
        yix0, yix1 = max(yix - ixrad[0], 0), min(yix + ixrad[0] + 1, rasterDim[0])
        xix0, xix1 = max(xix - ixrad[1], 0), min(xix + ixrad[1] + 1, rasterDim[1])
        kyix0, kyix1 = max(ixrad[0] - yix, 0), kernelDims[0] - max(
            (yix + ixrad[0]) - (rasterDim[0] - 1), 0
        )
        kxix0, kxix1 = max(ixrad[1] - xix, 0), kernelDims[1] - max(
            (xix + ixrad[1]) - (rasterDim[1] - 1), 0
        )
        Z[yix0:yix1, xix0:xix1] += w * kernel[kyix0:kyix1, kxix0:kxix1]
    return X, Y, Z


class Indexer2D:
    def __init__(self, rowfirst, nX, nY, xdir, ydir, rangex, rangey):
        self.rowfirst=rowfirst 
        self.nX=nX
        self.nY=nY
        self.xdir=xdir 
        self.ydir=ydir
        # Determine coefficients for index function
        self.x_ix0 = 0 if xdir == 1 else nX - 1
        self.y_ix0 = 0 if ydir == 1 else nY - 1
        self.x_ixstep = 1 if rowfirst else nY
        self.y_ixstep = nX if rowfirst else 1
        # Determine coefficients for position2index
        self.dx = xdir*(rangex[1]-rangex[0])/nX
        self.dy = ydir*(rangey[1]-rangey[0])/nY
        self.offsetx = rangex[0] if xdir==1 else rangex[1]
        self.offsety = rangey[0] if ydir==1 else rangey[1]
        
    def __call__(self, i, j):
        return (self.y_ixstep * (self.y_ix0 + self.ydir * j) 
                + self.x_ixstep * (self.x_ix0 + self.xdir * i))

    def getIndex(self, v, dv, offset):
        return int(np.round((v - offset)/dv))

    def getXIndex(self, x):
        return self.getIndex(x, self.dx, self.offsetx)
    
    def getYIndex(self, y):
        return self.getIndex(y, self.dy, self.offsety)
        
    def getNeighs(self, xy, rx, ry):
        # Return indices of neighbors of point xy within ranges rx, ry
        x, y = xy
        # Index base
        ixx, ixy = self.getXIndex(x), self.getYIndex(y)
        # randes converted to inices
        nrx, nry = int(np.ceil(np.abs(rx/self.dx)))+1, int(np.ceil(np.abs(ry/self.dy)))+1
        # neigh index ranges
        r_ixx, r_ixy = np.arange(ixx-nrx, ixx+nrx+1), np.arange(ixy-nry, ixy+nry+1)
        r_ixx = r_ixx[(0 <= r_ixx) & (r_ixx < self.nX)]
        r_ixy = r_ixy[(0 <= r_ixy) & (r_ixy < self.nY)]
        ixs = np.array([self(i,j) for i,j in product(r_ixx, r_ixy)])
        return ixs
    
        
    def makeMatrix(self, weights):    
        # Copy weight values into matrix
        Z = np.zeros((self.nX, self.nY))
        for i in range(self.nX):
            for j in range(self.nY):
                ix_ij = self(i,j)
                Z[i, j] = weights[ix_ij]
        return Z
        
    
def detect_grid(coords):
    # Try to derive the grid dimensions and ranges
    # from the given coordinate array
    x, y = coords[:,0], coords[:,1]
    rangex, rangey = (min(x), max(x)), (min(y), max(y))
    if rangex[0] == rangex[1] or rangey[0] == rangey[1]:
        # No variation of coords in one direction
        return None, "No variation of coords in one direction"
    dx, dy = x[1:] - x[:-1], y[1:] - y[:-1]
    if dx[0] != 0 and dy[0] != 0:
        # The first step of a serialized grid should only
        # increase one coordinate
        return (
            None,
            "The first step of a serialized grid should only increase one coordinate ",
        )

    # Determine index order of serialization
    if dx[0] != 0 and dy[0] == 0:
        rowfirst = True
        unique_dy = set(np.round(h, 12) for h in set(dy) if h != 0.0)
        unique_dx = set(np.round(h, 12) for h in set(dx) if h != 0.0)
        # dx sometimes jumps back from max to zero
        maxdx = np.max(np.abs(unique_dx))
        unique_dx.difference_update((maxdx, -maxdx))
    else:
        rowfirst = False
        unique_dx = set(np.round(h, 12) for h in set(dx) if h != 0.0)
        unique_dy = set(np.round(h, 12) for h in set(dy) if h != 0.0)
        # dy sometimes jumps back from max to zero
        maxdy = np.max(np.abs(list(unique_dy)))
        unique_dy.difference_update((maxdy, -maxdy))

    # Find number of grid points on axes
    if len(unique_dx) > 1:
        # Variable distances of steps in x-direction
        return None, "Variable distances of steps in x-direction"
    if len(unique_dy) > 1:
        # Variable distances of steps in y-direction
        return None, "Variable distances of steps in y-direction"

    dx0, dy0 = unique_dx.pop(), unique_dy.pop()

    # Determine number of grid points on axes
    X, Y = sorted(set(x)), sorted(set(y))
    if len(X) * len(Y) != len(coords):
        return (
            None,
            "Total number of points is different from nr of x-coords times nr of y-coords",
        )
    
    nX, nY = len(X), len(Y)
    xdir, ydir = int(np.sign(dx0)), int(np.sign(dy0))
    indexer = Indexer2D(rowfirst, nX, nY, xdir, ydir, rangex, rangey)
    return X, Y, indexer
