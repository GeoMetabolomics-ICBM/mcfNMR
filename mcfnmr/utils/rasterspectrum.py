from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
import numpy as np
from mcfnmr.utils.geometry import RectRegion
from mcfnmr.utils.parsesvg import loadRegions


class RasterSpectrum(object):
    """
    This is a class to handle 2D raster spectrum data.
    """

    def __init__(self, fullData, header, regionDefs=None, denoising_pars={}, name=None):
        self.header = header
        self.fullData = fullData
        self.pointDimensions = fullData.shape
        self.FRanges = header["FRanges"]
        self.smooth = denoising_pars.get("smoothRadius", 0.0)
        self.name = name

        # print("Loaded spectrum data for '%s'"%self.substance)
        print("Original grid dimension: %s" % str(fullData.shape))

        self.pointArea = self.calculatePointArea()
        # Raster points represent bins from here on
        self.fullData *= self.pointArea

        # Defining regions allows to disregard polygonal parts of the spectrum
        # as unknown. This allows to exclude noise, e.g.
        if regionDefs is not None:
            self.regions = loadRegions(**regionDefs)
        else:
            self.regions = self.getInnerRegion()

        # Range to use for noise measurement (in relative extent to fullspectral data raster), see getNoiseStd()
        self.noiseRegionRangeF1 = [0, 0.25]
        self.noiseRegionRangeF2 = [0.75, 1.0]
        # filter noise and smooth (sets self.origNoiseStd and self.noiseStd)
        self.noiseFilteringAndSmoothing(**denoising_pars)

        self.scanNr = header.get("ExpCount", None)
        self.smoothingHistory = []

    def noiseFilteringAndSmoothing(self, noiseFilterLevel=None, smoothRadius=0.0):
        # Calculating noiseStd before smoothing
        self.noiseStd = self.getNoiseStd()
        self.origNoiseStd = self.noiseStd
        print("Noise Std before processing: %g" % self.noiseStd)
        sm = smoothRadius if smoothRadius > 0 else self.smooth
        if sm > 0.0:
            self.smoothData(sm)
        self.noiseStd = self.getNoiseStd()
        print("Noise Std after processing: %g" % self.noiseStd)

        s2n = np.max(self.fullData) / self.noiseStd
        if noiseFilterLevel is not None:
            origMass = self.getTotalAveragePointIntensity(positiveWeightOnly=True)
            # Noise filter as multiples of self.noiseStd
            self.deleteNoise(noiseFilterLevel)

        mass = self.getTotalAveragePointIntensity()
        print("   Total avg point intensity: %g" % mass)
        print("   Max point intensity: %g" % np.max(self.fullData))
        if noiseFilterLevel is not None:
            print(
                "   Noise filter level: %d (corresponds to minimal signal intensity %g)"
                % (noiseFilterLevel, self.origNoiseStd * noiseFilterLevel)
            )
            print(
                "   Total cleared signal by noise filter: %g%s"
                % (100 * (1 - mass / origMass), "%")
            )
            print("    (Original noise std: %g)" % self.origNoiseStd)
            print("    (Original signal to noise: %g)" % s2n)
        else:
            print("   Noise std: %g" % self.noiseStd)
            print("   Signal to noise: %g" % s2n)
        print("")

    def isPointSpectrum(self):
        return False

    def getInnerRegion(self):
        ranges = self.FRanges
        coords = (
            (ranges[0][0], ranges[1][0]),
            (ranges[0][0], ranges[1][1]),
            (ranges[0][1], ranges[1][1]),
            (ranges[0][1], ranges[1][0]),
            (ranges[0][0], ranges[1][0]),
        )
        inner = Polygon(coords)
        return [inner]

    def getOuterRegions(self):
        # Construct regions covering the outside of the given ranges.
        # This is the minimal unknown region for the spectrum
        ranges = self.FRanges
        right = RectRegion((ranges[0][1], np.inf), (-np.inf, np.inf))
        left = RectRegion((-np.inf, ranges[0][0]), (-np.inf, np.inf))
        top = RectRegion((-np.inf, np.inf), (ranges[1][1], np.inf))
        bottom = RectRegion((-np.inf, np.inf), (-np.inf, ranges[1][0]))
        regions = [right, left, top, bottom]
        return regions

    def calculatePointArea(self):
        ranges, dimensions = self.FRanges, self.pointDimensions
        pointArea = (
            (ranges[0][-1] - ranges[0][0])
            * (ranges[1][-1] - ranges[1][0])
            / (dimensions[0] * dimensions[1])
        )
        return pointArea

    def smoothData(self, smoothRadius):
        # Apply a smoothing to the matrix
        print("Smoothing data with radius %g." % smoothRadius)
        self.fullData = gaussian_filter(
            self.fullData, sigma=smoothRadius, mode="mirror"
        )

    def getNoiseRegionData(self):
        ix11 = int(self.pointDimensions[0] * self.noiseRegionRangeF2[0])
        ix12 = int(self.pointDimensions[0] * self.noiseRegionRangeF2[1])
        ix21 = int(self.pointDimensions[1] * self.noiseRegionRangeF1[0])
        ix22 = int(self.pointDimensions[1] * self.noiseRegionRangeF1[1])
        return self.fullData[ix11:ix12, ix21:ix22].flatten()

    def getNoiseStd(self, useNegative=True):
        # Sample noise from corner of the spectrum
        noise = self.getNoiseRegionData()
        # Use negative part to estimate the std (avoids artifacts from peaks in the region,
        # although the selected region shouldn't hold any)
        if useNegative:
            noise = noise[noise <= 0]
            noise = np.concatenate((noise, -noise))
        noise_std = np.std(noise)
        return noise_std

    def deleteNoise(self, level):
        if self.fullData is None:
            print("Error at deleteNoise(): No data loaded.")
        else:
            assert level is not None
            self.fullData[self.fullData <= level * self.noiseStd] = 0
            self.noiseFiltered = True
            self.noiseStd = None

    def getRasterDataAsPeaks(
        self, nbin=(1, 1), xVar="F1", signalThreshold=None, cutGridToBin=True
    ):
        """
        This returns peak-coords and corresponding weights for the full grid intensity data without
        peak extraction. Returned coords are always the centers of the rasterpoints
        nbin - side length in raster points for binning
        signalThreshold - threshold to account for peaks relative to noise-std of the spectrum
        cutGridToBin - discard raster points if binning multiple does not match original raster point dimension
        """
        assert min(nbin) >= 1
        print(
            "Converting raster to peaks, nbin=%s, signal-to-noise threshold=%s)"
            % (nbin, str(signalThreshold))
        )

        assert self.FRanges[0][1] > self.FRanges[0][0]
        assert self.FRanges[1][1] > self.FRanges[1][0]

        nCutoff = (self.pointDimensions[0] % nbin[0], self.pointDimensions[1] % nbin[1])
        if not cutGridToBin:
            assert nCutoff[0] == 0
            assert nCutoff[1] == 0
            FRanges = self.FRanges
            ptDim = self.pointDimensions
            fullData = self.fullData
        else:
            # Grid point distances in full grid
            dF1 = (self.FRanges[0][1] - self.FRanges[0][0]) / self.pointDimensions[0]
            dF2 = (self.FRanges[1][1] - self.FRanges[1][0]) / self.pointDimensions[1]
            # Truncated dimensions (dividable by nbin)
            ptDim = [
                self.pointDimensions[0] - nCutoff[0],
                self.pointDimensions[1] - nCutoff[1],
            ]
            dCut = [nCutoff[0] * dF1, nCutoff[1] * dF2]
            FRanges = [list(self.FRanges[0]), list(self.FRanges[1])]
            fullData = self.fullData
            if FRanges[0][0] + dCut[0] <= 0.0:
                # cut left side of spectrum
                FRanges[0][0] += dCut[0]
                fullData = fullData[nCutoff[0] :, :]
            elif nCutoff[0] > 0:
                # cut right side of spectrum
                FRanges[0][1] -= dCut[0]
                fullData = fullData[: -nCutoff[0], :]
            if FRanges[1][0] + dCut[1] <= 0.0:
                # cut lower side of spectrum
                FRanges[1][0] += dCut[1]
                fullData = fullData[:, nCutoff[1] :]
            elif nCutoff[1] > 0:
                # cut upper side of spectrum
                FRanges[1][1] -= dCut[1]
                fullData = fullData[:, : -nCutoff[1]]

        if signalThreshold is not None:
            if self.noiseStd is None:
                noiseStd = self.getNoiseStd()
            else:
                noiseStd = self.noiseStd
            noiseLevel = noiseStd * signalThreshold
            print(
                "signal-to-noise threshold %g corresponds to intensity %g"
                % (signalThreshold, noiseLevel)
            )
            totalCleared = np.sum(fullData[fullData < noiseLevel])
            print("Clearing total intensity of %g" % (totalCleared))
            fullData[fullData < noiseLevel] = 0.0

        # Grid point distances if pointDimension % nbin == 0
        dF1 = (FRanges[0][1] - FRanges[0][0]) * nbin[0] / ptDim[0]
        dF2 = (FRanges[1][1] - FRanges[1][0]) * nbin[1] / ptDim[1]

        F10 = FRanges[0][0] + 0.5 * dF1
        F20 = FRanges[1][0] + 0.5 * dF2

        gridShape = (int(ptDim[0] / nbin[0]), int(ptDim[1] / nbin[1]))
        integralWeights = np.zeros(gridShape)
        integralCoords = np.zeros(list(gridShape) + [2])
        pointClusterArea = self.pointArea * nbin[0] * nbin[1]

        for i in range(gridShape[0]):
            for j in range(gridShape[1]):
                pointClusterMeanIntensity = (
                    np.mean(
                        fullData[
                            i * nbin[0] : (i + 1) * nbin[0],
                            j * nbin[1] : (j + 1) * nbin[1],
                        ]
                    )
                    / self.pointArea
                )
                integralWeights[i, j] = pointClusterMeanIntensity * pointClusterArea
                # # TODO: replace the above by equivalent (will cause numerical precision errors):
                # integralWeights[i, j] = np.sum(
                #         fullData[
                #             i * nbin[0] : (i + 1) * nbin[0],
                #             j * nbin[1] : (j + 1) * nbin[1],
                #         ]
                #     )
                if xVar == "F1":
                    integralCoords[i, j, :] = F10 + i * dF1, F20 + j * dF2
                else:
                    integralCoords[i, j, :] = F20 + j * dF2, F10 + i * dF1

        npts = integralWeights.size
        coords, weights = integralCoords.reshape((npts, 2)), integralWeights.reshape(
            (npts,)
        )
        print("   (original raster dimensions: %s)" % str(self.pointDimensions))
        print("   (binned grid dimensions: %s)" % str(gridShape))
        return coords, weights, pointClusterArea

    def getTotalAveragePointIntensity(
        self, positiveRangeOnly=False, positiveWeightOnly=False
    ):
        loaded = self.fullData is not None
        if not loaded:
            self.load()

        if positiveWeightOnly:
            data = self.fullData
            data[data < 0] = 0
        else:
            data = self.fullData
        if positiveRangeOnly:
            xspan = np.linspace(
                self.FRanges[1][0], self.FRanges[1][1], self.pointDimensions[1]
            )
            yspan = np.linspace(
                self.FRanges[0][0], self.FRanges[0][1], self.pointDimensions[0]
            )
            xiPos, yiPos = np.argmax(xspan > 0), np.argmax(yspan > 0)
            res = np.sum(data[yiPos:, xiPos:]) / (len(xspan) * len(yspan))
        else:
            res = np.sum(data)

        if not loaded:
            self.unload()
        return res


def sampleBlankSpectrum(rasterSpec, seed):
    # Use noise region of rasterSpec to generate a blank Spectrum
    # With the same noise distribution
    noise = rasterSpec.getNoiseRegionData().flatten()
    dims = rasterSpec.pointDimensions
    rng = np.random.default_rng(seed)
    fullData = rng.choice(noise.flatten(), replace=True, size=dims)
    blank = RasterSpectrum(
        fullData,
        header=rasterSpec.header,
        regionDefs=None,
        name=rasterSpec.name + "(blank, seed%d)" % seed,
    )
    blank.regions = rasterSpec.regions
    return blank
