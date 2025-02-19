import os
import numpy as np
import svglib.svglib as svg
import reportlab.graphics
from shapely.geometry import (
    Polygon,
    MultiPolygon,
)


def getAllShapes(contentList, lvl=0, verb=False):
    allShapes = []
    for c in contentList:
        if type(c) == reportlab.graphics.shapes.Group:
            shapes = getAllShapes(c.getContents(), lvl + 1, verb=verb)
            allShapes.extend(shapes)
        elif type(c) == reportlab.graphics.shapes.Path:
            if verb:
                print("Found Path ('%s') at depth %d." % (str(c), lvl))
            allShapes.append(c)
        elif type(c) == reportlab.graphics.shapes.Rect:
            if verb:
                print("Found Rect ('%s') at depth %d." % (str(c), lvl))
            allShapes.append(c)
        else:
            if verb:
                print("Found entry of type '%s' at depth %d." % (type(c), lvl))
    return allShapes


# drawing operators (probably used in reportlab)
operatorIDs = {"START": 0, "CONNECT": 1, "CLOSE": 3}


def makePolys(shapes, verb=0):
    polys = []
    for i, p in enumerate(shapes):
        if type(p) == reportlab.graphics.shapes.Rect:
            x, y, w, h = p.x, p.y, p.width, p.height
            pts = [(x, y), (x, y + h), (x + w, y + h), (x + w, y)]
        elif type(p) == reportlab.graphics.shapes.Path:
            pts, ops = p.points, p.operators
            if ops[-1] != operatorIDs["CLOSE"]:
                # No closed segment
                if verb > 0:
                    print("Path %d (%s) is not closed." % (i, p))
                continue
            xCoords = np.array(pts)[0::2]
            yCoords = np.array(pts)[1::2]
            pts = list(zip(xCoords, yCoords))
            if len(pts) < 3:
                if verb > 0:
                    print("Path %d (%s) only consists of %d points." % (i, p, len(pts)))
                continue
        if verb > 1:
            print("Adding polygon for shape %d (%s)" % (i, p))
        poly = Polygon(pts)
        polys.append(poly)
    if verb > 0:
        print("Generated %d polygons" % len(polys))
    return polys


def separateReferenceRect(polys):
    # Find poly with largest area to identify full area range
    areas = [p.area for p in polys]
    boundsIx = np.argmax(areas)
    bounds = polys[boundsIx]
    coords = bounds.exterior.coords
    # Test if fullRange is a rectangle
    errmsg = "Reference area must be \n   (1) largest area, \n   (2) rectangular"
    if len(coords) != 5:
        print(
            "Rectangle bounding area must have 4 points (+1 repeating the first to close it in shapely)"
        )
        raise Exception(errmsg)
    xcoords, ycoords = coords.xy
    minx, maxx, miny, maxy = min(xcoords), max(xcoords), min(ycoords), max(ycoords)
    boundsBB = bounds.bounds
    if (
        (boundsBB != (minx, miny, maxx, maxy))
        or (len(set(xcoords)) != 2)
        or (len(set(ycoords)) != 2)
    ):
        print(
            "Rectangle bounding area must not be rotated and rectangular (equal to its bounding box)"
        )
        raise Exception(errmsg)
    return bounds, [p for i, p in enumerate(polys) if i != boundsIx]


def normalizeCoords(polys, bounds):
    # Transform everything to the range [0,1]x[0,1]
    bb = bounds.bounds
    x0, y0 = bb[0], bb[1]
    scalex, scaley = bb[2] - bb[0], bb[3] - bb[1]
    transform = lambda x, y: ((x - x0) / scalex, (y - y0) / scaley)
    transformedPolys = []
    for p in polys:
        xcoords, ycoords = p.exterior.coords.xy
        transformedPts = [transform(x, y) for x, y in zip(xcoords, ycoords)]
        transformedPoly = Polygon(transformedPts)
        transformedPolys.append(transformedPoly)
    transformedBounds = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    return transformedPolys, transformedBounds


def scaleToRange(polys, ranges):
    # This expects polys in a normalized range:
    for p in polys:
        assert max(p.bounds) <= 1
        assert min(p.bounds) >= 0
    # Whether x and y regions are inverted
    invertX = ranges["x"][0] > ranges["x"][1]
    invertY = ranges["y"][0] < ranges["y"][1]  # svg has growing coords to bottom
    x0, x1 = min(ranges["x"]), max(ranges["x"])
    y0, y1 = min(ranges["y"]), max(ranges["y"])
    rx, ry = x1 - x0, y1 - y0

    if invertX:
        transformX = lambda x: x0 + rx * (1.0 - x)
    else:
        transformX = lambda x: x0 + rx * x
    if invertY:
        transformY = lambda y: y0 + ry * (1.0 - y)
    else:
        transformY = lambda y: y0 + ry * y

    transform = lambda xy: (transformX(xy[0]), transformY(xy[1]))

    transformed = []
    for i, p in enumerate(polys):
        shell, holes = [], []
        xCoords, yCoords = p.exterior.coords.xy
        for pt in zip(xCoords, yCoords):
            shell.append(transform(pt))
        for pp in p.interiors:
            hole = []
            xCoords, yCoords = pp.xy
            for pt in zip(xCoords, yCoords):
                hole.append(transform(pt))
            holes.append(hole)
        transformed.append(Polygon(shell, holes))
    return transformed


def invertPolys(polys, bounds, verb=False):
    # Take the complement of each poly within bounds
    inverted = []
    for i, p in enumerate(polys):
        if verb:
            print("Polygon %d:\n%s" % (i, p))
        complement = bounds.difference(p)
        if verb:
            print(
                "Complement of %s %d is %s"
                % (
                    str(type(p)).split(".")[-1][:-2],
                    i,
                    str(type(complement)).split(".")[-1][:-2],
                )
            )
        if type(complement) == MultiPolygon:
            geoms = complement.geoms
            for j, pp in enumerate(geoms):
                if verb:
                    print("   complement part %d.exterior: %s" % (j, pp.exterior))
                    print(
                        "   complement part %d.interiors: %s" % (j, list(pp.interiors))
                    )
                assert type(pp) == Polygon
                inverted.append(pp)
        elif type(complement) == Polygon:
            inverted.append(complement)
            if verb:
                print("  complement.exterior:", complement.exterior)
                print("  complement.interiors:", list(complement.interiors))
        else:
            print("Unexpected type of complement!")
            raise Exception("Unexpected type of complement: %s" % type(complement))
    return inverted


def mergePolys(polys, verb=False):
    mergedPolys = []
    for i, poly in enumerate(polys):
        # NOTE: In the following, we assume that no higher
        # nesting of polygons, i.e., only single holes occur
        # Further we assume that any holes do only pierce one
        # containing polygon
        containment = False
        for j in range(len(mergedPolys)):
            p = mergedPolys[j]
            if p.contains(poly):
                if verb:
                    print(
                        "Polygon %d is contained in merged polygon %d (extend of smaller: %s)"
                        % (i, j, poly.bounds)
                    )
                assert len(poly.interiors) == 0
                mergedPolys[j] = Polygon(
                    p.exterior, holes=list(p.interiors) + [poly.exterior]
                )
                containment = True
                break
            elif p.within(poly):
                if verb:
                    print(
                        "Polygon %d contains merged polygon %d (extend of smaller: %s)"
                        % (i, j, p.bounds)
                    )
                assert len(p.interiors) == 0
                mergedPolys[j] = Polygon(poly.exterior, holes=[p.exterior])
                containment = True
                break
        if not containment:
            # No containment relation with any existing
            if verb:
                print(
                    "New polygon %d disjoint from all previous. Adding it to collection."
                    % (i)
                )
            mergedPolys.append(poly)

    # Test: to check validity (no intersections of elements), make collection
    MultiPolygon(mergedPolys)

    return mergedPolys


def cropAndDiscardUncontained(polys, bounds):
    contained = []
    for i, p in enumerate(polys):
        if not p.intersection(bounds):
            print(
                "Polygon %d doesn't intersect with bounding rectangle => discarding\n   (%s)"
                % (i, p)
            )
        else:
            cropped = p.intersection(bounds)
            assert type(cropped) == Polygon
            contained.append(cropped)
    return contained


def loadRegions(
    svgFile, originalRanges, invertRegions=False, flipAxes=False, verb=True
):
    if not os.path.exists(svgFile):
        return []
    drawing = svg.svg2rlg(svgFile)
    if drawing is None:
        print("Couldn't parse regions file '%s'" % svgFile)
    allShapes = getAllShapes(drawing.getContents())
    polys = makePolys(allShapes)
    bounds, polys = separateReferenceRect(polys)
    polys, bounds = normalizeCoords(polys, bounds)
    polys = cropAndDiscardUncontained(polys, bounds)
    polys = mergePolys(polys)
    if invertRegions:
        polys = invertPolys(polys, bounds)
    polys = scaleToRange(polys, originalRanges)
    if flipAxes:
        polys = swapAxes(polys)
    if verb:
        print("Loaded regions:")
        for i, p in enumerate(polys):
            print("   Polygon %d: %s" % (i, p))
    return polys


def swapAxes(polys):
    swapped = []
    for p in polys:
        newShell = list(zip(p.exterior.xy[1], p.exterior.xy[0]))
        newHoles = []
        for i in p.interiors:
            newHoles.append(list(zip((i.xy[1], i.xy[0]))))
        swapped.append(Polygon(shell=newShell, holes=newHoles))
    return swapped


def test1():
    ## Input
    # Source file for shapes
    svgfn = "/home/leo/Workspace/NMR_analysis/data/full spectra/DOM/HSQC_NELHA.007.001.2rr_processed.svg"
    # Whether to ignore the complements of the provided regions, rather
    invertRegions = True
    # Ranges of the underlying data (left-to-right on x-axis and bottom-to-top on y-axis)
    ranges = {"x": (9.5, -2.5), "y": (150, 0)}

    drawing = svg.svg2rlg(svgfn)
    contents = drawing.getContents()
    allShapes = getAllShapes(drawing.getContents(), 0)
    polys = makePolys(allShapes)
    bounds, polys = separateReferenceRect(polys)
    polys, bounds = normalizeCoords(polys, bounds)

    # Test cases:
    # (1) discard outside bounds
    polyOutsideBounds = Polygon([(1000, 1000), (1000, 1001), (1001, 1001)])
    # (2) handle poly which exceeds bounding box range
    # polyDisectingBounds = Polygon([(-0.1, -0.1), (1.1,1.1), (1.2,1.1)])
    polyDisectingBounds = Polygon([(0.4, -0.1), (0.4, 1.1), (0.6, 1.1), (0.6, -0.1)])
    # (3) handle poly which lies within another
    polyWithinOther = Polygon([(0.6, 0.1), (0.65, 0.2), (0.6, 0.2)])
    # Add test cases
    polys.extend([polyOutsideBounds, polyDisectingBounds, polyWithinOther])

    polys = cropAndDiscardUncontained(polys, bounds)
    polys = mergePolys(polys)
    if invertRegions:
        polys = invertPolys(polys, bounds)
    polys = scaleToRange(polys, ranges)

    for i, p in enumerate(polys):
        assert type(p) == Polygon
        print("Polygon %d: %s" % (i, p))


def test2():
    # Source file for shapes
    svgFile = "/home/leo/Workspace/NMR_analysis/data/full spectra/DOM/HSQC_NELHA.007.001.2rr_processed.svg"
    # Whether to ignore the complements of the provided regions, rather
    invertRegions = False
    # Ranges of the underlying data (left-to-right on x-axis and bottom-to-top on y-axis)
    originalRanges = {"x": (9.5, -2.5), "y": (150, 0)}

    loadRegions(svgFile, originalRanges, invertRegions)


if __name__ == "__main__":
    test2()
