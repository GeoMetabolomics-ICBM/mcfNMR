import os
import pickle
from timeit import default_timer
import numpy as np
import networkx as nx
from scipy.optimize import linprog
from scipy.sparse import lil_array

from mcfnmr.config import (
    DEFAULT_SCALE_C,
    DEFAULT_SCALE_H,
    MAX_ABSORPTON_COST,
    OUTDIR,
    DEBUG,
)
from mcfnmr.utils.geometry import load_regions, getIndicesWithin
from mcfnmr.utils.pointspectrum import detect_grid


class MCFResult(object):
    def __init__(self, total_target_weight, stats, absorption_cost, compounds, timings):
        self.originalWeightY = total_target_weight
        self.absorptionCost = absorption_cost
        self.timings = timings
        self.compounds = compounds
        compoundIDs = sorted(compounds)

        self.nPeaksConsidered = stats.get("nPeaksConsidered")
        self.inflowsY = stats.get("inflowsY")
        self.residualsY = stats.get("residualsY")
        self.residualsX = stats.get("residualsX")
        self.assigned = stats.get("flowsC2Y")
        total_weights = {
            c: compounds[c]["total_weight"] if compounds[c]["total_weight"] > 0 else 1.0
            for c in compoundIDs
        }
        self.assigned_relative = {
            c: self.assigned[i] / total_weights[c] for i, c in enumerate(compoundIDs)
        }
        self.assignmentCosts = stats.get("costsC2Y")
        self.unknownFlow = stats.get("unknownFlowC")
        self.specificCosts = stats.get("specificCostsC")
        self.inflowsC = stats.get("inflowsC")
        self.absorbedFlow = stats.get("absorbedFlow")
        self.totalAbsorption = np.sum(self.absorbedFlow)
        self.totalAssignedFlow = np.sum(self.assigned)
        self.totalAbsorptionCost = self.totalAbsorption * absorption_cost
        self.totalAssignmentCost = sum(self.assignmentCosts)
        self.totalCost = self.totalAbsorptionCost + self.totalAssignmentCost

        print("Total flows:")
        print("   absorbed: %g" % self.totalAbsorption)
        print("   assigned: %g" % self.totalAssignedFlow)
        print("Total costs:")
        print(" assignment: %g" % self.totalAssignmentCost)
        print(" absorption: %g" % self.totalAbsorptionCost)
        print("      total: %g" % self.totalCost)


def serialize(lib):
    compounds = sorted(lib.keys())
    ixs, all_weights, all_coords = {}, [], []
    current_ix = 0
    for compound_name in compounds:
        spectrum = lib[compound_name]
        if np.any(spectrum.weights < 0):
            raise Exception(f"Negative weight for '{spectrum.name}'")
        ixs[compound_name] = np.arange(current_ix, current_ix + len(spectrum.weights))
        all_weights.extend(spectrum.weights)
        all_coords.extend(spectrum.coords)
        current_ix += len(spectrum.weights)
    return ixs, np.array(all_weights), np.array(all_coords)


def euclidean_dists(xCoords, yCoords, scales, assignment_radius):
    """
    Returns two lists (distList, indexList) of length nx containing distances
    and corresponding indices of y-coords, where x and y distances are weighted
    according to the given scalings. Only points within the given assignment radius
    are included into the lists.
    """
    assert yCoords.shape[1] == xCoords.shape[1]
    nx, ny, ndim = xCoords.shape[0], yCoords.shape[0], xCoords.shape[1]
    assert len(scales) == ndim
    
    # Try to deduce whether target is a grid spectrum
    res = detect_grid(yCoords)
    griddedY = res[0] is not None
    if DEBUG and not griddedY:
        print(f"Could not deduce grid for spectrum.\n   ({res[1]})")
    else:
        _, _, indexer = res
    
    distList, indexList = [], []
    for i in range(nx):
        x = xCoords[i]
        if griddedY:
            # Determine indices in rectangle  corresponding to quadrat of side 
            # length assignment_radius in rescaled coords
            rx, ry = assignment_radius*scales[0], assignment_radius*scales[1]
            # xspan, yspan indices
            ixs = indexer.getNeighs(x, rx, ry)
            
            if DEBUG > 1:
                print(f"\nNeighs of point {x} in radius {assignment_radius:g}:")
                for ix in ixs:
                    print(f"At {ix}:", yCoords[ix])
        else:
            # Calculate dists for each target peak if 
            # target is no grid spectrum
            ixs = np.arange(ny, dtype=int)
        
        a = np.zeros(len(ixs))
        if len(ixs) > 0:
            for k in range(ndim):
                a += np.power((yCoords[ixs, k] - x[k]) / scales[k], 2)
        a = np.sqrt(a)
        ix = np.where(a <= assignment_radius)[0]
        distList.append(a[ix])
        indexList.append(ixs[ix])
        
    if DEBUG:
        print("len(distList):", sum(len(l) for l in distList))
    return distList, indexList


def makeMCFNet(
    dist_info,
    weightsY,
    compound_ix,
    weightsX,
    absorption_cost,
    isolated_fit=False,
    ixYinY=None,
    ixXinY=None,
):
    """
    Construct the flow network to route a unit mass through compound nodes C to peak layer X,
    respecting given proportionality constraints, and further to Y with minimal costs.

    dist_info:
        A tuple consisting of a list of distances of X- and Y-nodes
        (i-th element==entries for Y-nodes paired with the i-th X-node)
        and the corresponding indices. Determines the connectivity of the network.
    weightsY:
        capacities of nodes in the target layer
    compound_ix:
        indices of the different compounds in weightsX
    weightsX:
        a list of node weights for the compounds (specified by compound_ix)
    absorption_cost:
        specific cost of absorbed flow
    isolated_fit:
        whether all compounds should be fitted independently. Otherwise a joint fit
        is executed, which avoids overassignment of flow to target nodes.
    ixYinY, ixXinY:
        Boolean arrays, which may exclude indices from matching (mainly used for
        disregarding compound peaks outside the target area, but could also be used
        if match is performed on a restricted range)

    The resulting network has a node indexing which assigns nodes as follows:
    X-nodes: ixs in [0,...,nx-1] with capacities
    Y-nodes (sinks): ixs in [nx,...,nx+ny-1]
    Compound nodes: ix = nx+ny+i, i=0,...,nCompounds-1
    absorbing node (sink): ix = nx+ny+nCompounds
    if not isolated_fit:
        Source node: ix = nx+ny+nCompounds+1
    else:
        Compound nodes are sources

    The layered structure is
    source -> compounds -> X-peaks -> Y-peaks
    """
    assert min(weightsY) >= 0
    assert min(weightsX) > 0

    if ixYinY is None:
        ixYinY = np.arange(len(weightsY))
    if ixXinY is None:
        ixXinY = np.arange(len(weightsX))
    weightsX[~ixXinY] = 0.0
    weightsY[~ixYinY] = 0.0

    compoundIDs = sorted(compound_ix.keys())
    source_flow = sum(weightsY[ixYinY])

    dist_list, index_list = dist_info

    nX, nY, nC = (
        sum([len(ix) for c, ix in compound_ix.items()]),
        len(weightsY),
        len(compound_ix),
    )
    # Y-node and compound-node indexing
    yix = lambda i: nX + i
    cix = lambda i: nX + nY + i
    # Absorbing, ignoring and source nodes
    aix = nX + nY + nC
    # global source (only needed if not isolatedCompoundFit)
    six = nX + nY + nC + 1

    # For convenience, make a dict containing for each compound
    # the node indices and the proportions of its peaks
    compounds = dict()
    for c, ix in compound_ix.items():
        inY = [ixXinY[j] for j in ix]
        if not np.any(inY):
            print(f"No peak of {c} in considered region.")
            compounds[c] = dict(ix=[], p=[], total_weight=0.0)
            continue
        total_weight = sum(weightsX[ix[inY]])
        compounds[c] = dict(
            ix=ix[inY], p=weightsX[ix[inY]] / total_weight, total_weight=total_weight
        )

    # Make the flownet digraph
    G = nx.DiGraph(nX=nX, nY=nY, compounds=compounds, isolated_fit=isolated_fit)
    emptyCompounds = []
    xnodes, cnodes = [], []
    for i, c in enumerate(compoundIDs):
        cInfo = compounds[c]
        if sum(cInfo["p"]) == 0:
            emptyCompounds.append(i)
            continue
        # Weights on X-nodes are proportions wrt compound for X-peaks
        xnodes.extend(
            [(n, {"w": w}) for n, w in zip(cInfo["ix"], cInfo["p"]) if ixXinY[n]]
        )
        if isolated_fit:
            w = source_flow
        else:
            w = 0.0
        cnodes.append((cix(i), {"w": w}))

    G.add_nodes_from(cnodes)
    G.add_nodes_from(xnodes)

    # Ensure that absorbing nodes has sufficient capacity to
    # take up all flow from sources if necessary
    total_source = (
        (nC - len(emptyCompounds)) * source_flow if isolated_fit else source_flow
    )
    G.add_node(aix, w=total_source + 1)

    if not isolated_fit:
        G.add_node(six, w=source_flow)
        # Construct edges source->compounds
        s2cEdges = [
            (six, cix(i)) for i, c in enumerate(compoundIDs) if not c in emptyCompounds
        ]
        G.add_edges_from(s2cEdges)

    # Register X-peaks without Y-neighs
    noNeighs = []
    totalTargetWeight = {}

    for i, c in enumerate(compoundIDs):
        if c in emptyCompounds:
            continue
        cInfo = compounds[c]
        # Construct edges compound -> X-peaks
        c2XEdges = [(cix(i), j) for j, w in zip(cInfo["ix"], cInfo["p"]) if ixXinY[j]]
        G.add_edges_from(c2XEdges)

        if DEBUG > 0:
            totalTargetWeight[c] = 0.0

        for j in cInfo["ix"]:
            if j not in G.nodes:
                noNeighs.append(j)
                continue
            # Construct edges X-peaks -> Y-peaks
            dists, ixs = dist_list[j], index_list[j]
            X2YEdges = [
                (j, yix(ix), w) for ix, w in zip(ixs, dists) if ixXinY[j] and ixYinY[ix]
            ]
            if X2YEdges:
                G.add_weighted_edges_from(X2YEdges)
                if DEBUG > 0:
                    targetWeight_j = sum([weightsY[ixy - nX] for _, ixy, _ in X2YEdges])
                    totalTargetWeight[c] += targetWeight_j
            else:
                if DEBUG > 1:
                    print(
                        "No Y-nodes in assignment neighborhood for X-node %d (x in Y: %s)"
                        % (j, ixXinY[j])
                    )
                noNeighs.append(j)
        if DEBUG > 1:
            print(
                f"\nTotal connected target weight for '{c}': %g" % totalTargetWeight[c]
            )

    # Add weight to ynodes
    nrY = 0
    for n, w in enumerate(weightsY):
        if yix(n) in G.nodes:
            G.nodes[yix(n)]["w"] = w
            nrY += 1
    # Debug
    if DEBUG > 0:
        print("\nTotal connected target nodes: %d" % nrY)
        print("Total target weight: %g\n" % sum(weightsY))

    # Add edge from sources to absorbing node
    if isolated_fit:
        absorptionEdges = [
            (cix(i), aix, absorption_cost)
            for i, c in enumerate(compoundIDs)
            if not c in emptyCompounds
        ]
    else:
        absorptionEdges = [(six, aix, absorption_cost)]
    G.add_weighted_edges_from(absorptionEdges)

    if DEBUG > 1:
        for i in range(nX):
            if i in G.nodes:
                print("       Node %d: %s" % (i, G.nodes[i]))
                print("         preds: %s" % (list(G.predecessors(i))))
                print("         succs: %s" % (list(G.successors(i))))
                if len(list(G.successors(i))) == 0:
                    print("No successors for node %d" % i)
                print(
                    "  w/capacities: %s"
                    % (["%.5f" % G.nodes[n]["w"] for n in G.successors(i)])
                )
            else:
                print("Node %d not in G" % (i))
        if not isolated_fit:
            print("Source node %d:" % six)
            print("   succs: %s" % (list(G.successors(six))))
        print("emptyCompounds: %s" % str(emptyCompounds))
        print("Number of isolated X-nodes: %d" % len(noNeighs))
    return G


def calculateMCF(flownet):
    """
    Solve the linear program, which gives the min cost flow defined by the
    given network with compound nodes
    """
    isolated_fit = flownet.graph["isolated_fit"]
    compounds = flownet.graph["compounds"]
    nX = flownet.graph["nX"]
    nY = flownet.graph["nY"]
    nC = len(compounds)
    compoundIDs = sorted(compounds.keys())

    # Absorbing, ignoring and source nodes
    # Y-node and compound-node indexing
    yix = lambda i: nX + i
    cix = lambda i: nX + nY + i
    aix = nX + nY + nC
    if not isolated_fit:
        six = nX + nY + nC + 1

    # Ordered edges:
    edges = list(flownet.edges())
    nE = len(edges)
    # Reverse map edge->index
    edgeIndices = {e: i for i, e in enumerate(edges)}

    print("Constructing A_eq...")
    tic = default_timer()
    if isolated_fit:
        nEq = 2 * nX + nC
    else:
        nEq = 2 * nX + nC + 1
    A_eq = lil_array((nEq, nE))
    b_eq = np.zeros(nEq)

    # (1) nX+nC(+1) feed-forward flow equalities:
    #     For each node from {X-peaks, compounds(, source)},
    #     the flow sum over all outgoing edges equals
    #     the production plus the incoming flow
    nEq = 0  # reset and use as counting index
    # x-nodes
    deadendC = set()

    for ix, c in enumerate(compoundIDs):
        cInfo = compounds[c]
        peakIndices = cInfo["ix"]

        connected = np.all(
            [
                len(list(flownet.successors(xix))) > 0
                for xix in peakIndices
                if xix in flownet.nodes
            ]
        ) and np.any([j in flownet.nodes for j in peakIndices])
        
        if not connected:
            # no successors → no inflow from compound node
            # Will be treated below adding a zero flow condition
            #    f(s→c) = 0
            # or a full absorption condition
            #    f(c→a) = weight(c)
            # for isolated fitting
            deadendC.add(ix)
            if DEBUG > 1:
                print(
                    f"Encountered dead end(s) at X layer, no assignment for compound {c}"
                )
            continue
        # For all peaks of c, ensure inflow==outflow
        for xix in peakIndices:
            if not xix in flownet.nodes:
                continue
            outEdgeIndices = [edgeIndices[(xix, s)] for s in flownet.successors(xix)]
            inEdgeIndices = [edgeIndices[(p, xix)] for p in flownet.predecessors(xix)]
            A_eq[nEq, inEdgeIndices] = -1.0
            A_eq[nEq, outEdgeIndices] = 1.0
            nEq += 1

    if DEBUG > 0:
        print("Number of dead end compounds: %d / %d" % (len(deadendC), nC))

    # Compound-nodes
    
    # NOTE: This ensures that the total outflow from c-node equals either
    #       its source production (isolated fit) or its inflow.
    #            sum[f(c→x_i)] = f(s→c) (or w_c).
    #       As the next constraints are
    #            f(c→x_i) = p_i*f(s→c) (or p_i*w_c),
    #       the above is implied by sum(p_i)=1 and hence redundant *if* c is 
    #       not empty, i.e. has associated peaks in the considered regions.
    #       if it is empty, we need to add f(s→c)=0 explicitly here. 
    
    if not isolated_fit:
        for i, c in enumerate(compoundIDs):
            cInfo = compounds[c]
            peakIndices = cInfo["ix"]
            if i not in deadendC:
                # Non-empty compound is captured by proportionality constraints below 
                continue
            ix = cix(i)
            if not ix in flownet.nodes:
                continue
            outEdgeIndices = [edgeIndices[(ix, s)] for s in flownet.successors(ix)]
            A_eq[nEq, outEdgeIndices] = 1.0
            inEdgeIndices = [edgeIndices[(p, ix)] for p in flownet.predecessors(ix)]
            A_eq[nEq, inEdgeIndices] = -1.0
            assert list(flownet.predecessors(ix))[0] == six
            nEq += 1
        else:
            # No need to specify outflow c == weight(c),
            # because this follows from the compoundwise
            # specifications, below (proportionality constraints)
            pass

    # Debug
    if DEBUG > 1:
        AA = A_eq.todense()[:nEq, :]
        rk = np.linalg.matrix_rank(AA)
        print("2: rank:%d, nEq:%d" % (rk, nEq))

    # if not isolated_fit (joint compound fit): add global source in last row of A_eq, below

    # (2) nX equalities: flow through compound nodes is proportionally directed to
    #                    the different peaks.
    #                    In case of a global source s, we have
    #                        f(c→xi) = pi*f(s→c) = pi*f(c→X)
    #                    If each c-node is a source with production p:
    #                        f(c→xi) = pi*(p - f(c→a)) = pi*f(c→X)
    #                    <=> f(c→xi) + pi*f(c→a) = pi*p
    for ix, c in enumerate(compoundIDs):
        cInfo = compounds[c]
        peakIndices = cInfo["ix"]
        c2xEdges = [(cix(ix), j) for j in peakIndices if j in flownet.nodes]
        c2xEdgeIndices = [edgeIndices[e] for e in c2xEdges]
        totalWeightC = sum(
            [p for j, p in zip(cInfo["ix"], cInfo["p"]) if j in flownet.nodes]
        )
        if totalWeightC == 0:
            continue
        if isolated_fit:
            c2aEdgeIx = edgeIndices[(cix(ix), aix)]
        else:
            s2cEdgeIx = edgeIndices[(six, cix(ix))]

        if ix in deadendC:
            # There are peaks without successors → no flow to any peak of c
            if isolated_fit:
                # All weight will be absorbed
                A_eq[nEq, c2aEdgeIx] = 1
                b_eq[nEq] = flownet.nodes[cix(ix)]["w"]
            else:
                # Inflow to c must be zero
                A_eq[nEq, s2cEdgeIx] = 1
            nEq += 1
            continue

        proportions = [cInfo["p"][i] / totalWeightC for i in range(len(peakIndices))]
        for p, eix in zip(proportions, c2xEdgeIndices):
            A_eq[nEq, eix] = 1
            if isolated_fit:
                b_eq[nEq] = p * flownet.nodes[cix(ix)]["w"]
                A_eq[nEq, c2aEdgeIx] = p
            else:
                A_eq[nEq, s2cEdgeIx] = -p
            nEq += 1

    # Debug
    if DEBUG > 1:
        AA = A_eq.todense()[:nEq, :]
        rk = np.linalg.matrix_rank(AA)
        print("3: rank:%d, nEq:%d" % (rk, nEq))

    # Add global source node
    if not isolated_fit:
        outEdgeIndices = [edgeIndices[(six, s)] for s in flownet.successors(six)]
        A_eq[nEq, outEdgeIndices] = 1.0
        b_eq[nEq] = flownet.nodes[six]["w"]
        nEq += 1

    # Debug
    if DEBUG > 0:
        cropCheckVal = np.sum(np.abs(A_eq[nEq:, :]))
        assert cropCheckVal == 0

    # Crop unneeded rows
    A_eq = A_eq[:nEq, :]
    b_eq = b_eq[:nEq]
    toc = default_timer()
    print("   took %g secs" % (toc - tic))

    # Debug
    if DEBUG > 1:
        AA = A_eq.todense()[:nEq, :]
        rk = np.linalg.matrix_rank(AA)
        print("4: rank:%d, nEq:%d" % (rk, nEq))

    # (3) Inequalities to account for the maximally matched target weights:
    print("Constructing A_ub...")
    tic = default_timer()
    if isolated_fit:
        # For isolated compound fit: Inflow to ynode from each compound is less than its capacity
        # In order to estimate the number of inequalities, we count
        # for each compound the number of connected Y-nodes
        connectedYNodes = {}
        for c in compoundIDs:
            cInfo = compounds[c]
            peakIndices = cInfo["ix"]
            connectedYNodes[c] = set()
            for xix in peakIndices:
                if xix not in flownet.nodes:
                    continue
                connectedYNodes[c].update(set(flownet.successors(xix)))
            print(
                "Compound %s has %d connected target nodes."
                % (c, len(connectedYNodes[c]))
            )
        nUb = sum([len(ns) for ns in connectedYNodes.values()])
        print("Total estimated number of inequalities is %d." % nUb)
        A_ub = lil_array((nUb, nE))
        b_ub = np.zeros(nUb)

        nUb = 0  # reset to determine the true number of inequalities
        for c in compoundIDs:
            cInfo = compounds[c]
            peakIndices = set(cInfo["ix"])
            for nj in connectedYNodes[c]:
                if DEBUG > 1:
                    if "w" not in flownet.nodes[nj]:
                        print("No weight for Y-Node %d!?" % (nj))
                b_ub[nUb] = max(0.0, flownet.nodes[nj]["w"])
                preds_c = set(flownet.predecessors(nj)).intersection(peakIndices)
                inEdgeIndices = [edgeIndices[(p, nj)] for p in preds_c]
                A_ub[nUb, inEdgeIndices] = 1.0
                nUb += 1
    else:
        # Joint compound fit: Inflow at each ynode is less than its capacity
        # Determine y-nodes with predecessors, because this is the number of
        # inequalities needed
        connectedYNodes = [
            yix(i)
            for i in range(nY)
            if yix(i) in flownet.nodes and len(flownet.pred[yix(i)]) > 0
        ]
        nUb = len(connectedYNodes)
        print(
            "Estimated number of inequalities: %d (nr of connected target nodes)." % nUb
        )
        A_ub = lil_array((nUb, nE))
        b_ub = np.zeros(nUb)
        nUb = 0  # reset to determine the true number of inequalities
        for nj in connectedYNodes:
            if DEBUG > 1:
                if "w" not in flownet.nodes[nj]:
                    print("No weight for Y-Node %d!?" % (nj))
            b_ub[nUb] = max(0.0, flownet.nodes[nj]["w"])
            inEdgeIndices = [edgeIndices[(p, nj)] for p in flownet.predecessors(nj)]
            A_ub[nUb, inEdgeIndices] = 1.0
            nUb += 1
    # regularRows = np.sum(A_ub,1) != 0

    if DEBUG > 0:
        print("True number of inequalities is %d." % nUb)
        cropCheckVal = np.sum(np.abs(A_ub[nUb:, :]))
        assert cropCheckVal == 0

    # Crop unneeded rows
    A_ub = A_ub[:nUb, :]
    b_ub = b_ub[:nUb]
    toc = default_timer()
    print("   took %g secs" % (toc - tic))

    # Check for

    # (3) Costvector
    edgeCosts = np.array([flownet.edges[e].get("weight", 0.0) for e in edges])
    inf_ix = edgeCosts == np.inf
    edgeCosts[inf_ix] = np.max(edgeCosts[np.logical_not(inf_ix)]) + 1000

    print("Running linprog()...")
    tic = default_timer()

    res = linprog(edgeCosts, A_ub, b_ub, A_eq, b_eq, options=dict(disp=True))

    toc = default_timer()
    print("   took %g secs" % (toc - tic))
    if DEBUG > 1:
        print("\nlinprog result:\n", res)
        print("\nlinprog.message:\n%s" % res.message)
    else:
        print("\nlinprog:\n%s" % res.message)

    optFlow = res.x
    for e, f in zip(edges, optFlow):
        flownet.edges[e]["flow"] = f


def flowstats(flownet, includeYStats=False):
    nX, nY = flownet.graph["nX"], flownet.graph["nY"]
    compounds = flownet.graph["compounds"]
    isolated_fit = flownet.graph["isolated_fit"]
    compoundIDs = sorted(compounds.keys())
    nC = len(compoundIDs)

    # Y-node and compound-node indexing
    yix = lambda i: nX + i
    cix = lambda i: nX + nY + i

    # Absorbing nodes
    aix = nX + nY + nC
    if not isolated_fit:
        six = nX + nY + nC + 1

    flowsC2Y = np.zeros(nC)
    costsC2Y, specificCostsC = np.zeros(nC), np.zeros(nC)
    nPeaksConsidered = np.zeros(nC)  # nr of peaks considered for the compound
    inflowsC = np.zeros(nC)
    flowsX2Y = np.zeros(nX)

    for i, c in enumerate(compoundIDs):
        if cix(i) in flownet:
            nPeaksInNet = len(list(flownet.successors(cix(i))))
        else:
            nPeaksInNet = 0
        nPeaksConsidered[i] = nPeaksInNet
        if DEBUG > 1:
            print("\n#Compound '%s'" % c)
        cInfo = compounds[c]
        flowsXC2Y, costsX2Y = [], []
        for n in cInfo["ix"]:
            if n not in flownet.nodes:
                continue
            succs = list(flownet.successors(n))
            flow_n = sum(
                [flownet.edges[(n, s)]["flow"] for s in succs if s not in {aix}]
            )
            flowsXC2Y.append(flow_n)
            flowsX2Y[n] = flow_n
            cost_n = sum(
                [
                    flownet.edges[(n, s)]["flow"] * flownet.edges[(n, s)]["weight"]
                    for s in succs
                    if s not in {aix}
                ]
            )
            costsX2Y.append(cost_n)
            specCost = cost_n / (flow_n) if flow_n != 0 else 0.0
            if DEBUG > 1:
                print(
                    "   Assigned flow from x-node %d (of compound '%s'): %g (specific cost: %g)"
                    % (n, c, flow_n, specCost)
                )
        flowsC2Y[i] = sum(flowsXC2Y)
        costsC2Y[i] = sum(costsX2Y)
        if cix(i) in flownet and not isolated_fit:
            inflowsC[i] = flownet.edges[(six, cix(i))]["flow"]
        else:
            inflowsC[i] = 0.0
        specificCostsC[i] = costsC2Y[i] / flowsC2Y[i] if flowsC2Y[i] != 0 else 0.0

    if isolated_fit:
        # Sources are compound nodes
        absorbedFlow = [
            flownet.edges[(cix(i), aix)]["flow"] if cix(i) in flownet else 0.0
            for i in range(nC)
        ]
        absorbedFlowTotal = sum(absorbedFlow)
    else:
        absorbedFlow = flownet.edges[(six, aix)]["flow"]
        absorbedFlowTotal = absorbedFlow

    if DEBUG > 0:
        print("Total absorbed flow: %g" % absorbedFlowTotal)
    if DEBUG > 1:
        for i, c in enumerate(compoundIDs):
            print("\nCompound %d: '%s'" % (i, c))
            print("  Total compound flow assigned: %g" % flowsC2Y[i])
            if isolated_fit:
                print("  Total absorption: %g" % (absorbedFlow[i]))
            print("  Total compound costs: %g" % costsC2Y[i])
            print("  Specific compound costs: %g" % specificCostsC[i])

    if includeYStats:
        inflowsY = np.zeros(nY)
        for n in range(nY):
            nj = yix(n)
            if nj not in flownet.nodes:
                continue
            preds = flownet.predecessors(nj)
            inflowsY[n] = sum([flownet.edges[(p, nj)]["flow"] for p in preds])
            if nY <= 20:
                print(
                    "Flow/capacity y-node %d: %g/%g"
                    % (n, inflowsY[n], flownet.nodes[nj]["w"])
                )
        residualsY = np.zeros(nY)
        for n in range(nY):
            nj = yix(n)
            v = flownet.nodes.get(nj, None)
            if v:
                residualsY[n] = v["w"] - inflowsY[n]
    else:
        inflowsY, residualsY = None, None

    stats = {
        "inflowsY": inflowsY,
        "inflowsC": inflowsC,
        "residualsY": residualsY,
        "flowsC2Y": flowsC2Y,
        "residualsX": None,
        "costsC2Y": costsC2Y,
        "absorbedFlow": absorbedFlow,
        "specificCostsC": specificCostsC,
        "nPeaksConsidered": nPeaksConsidered,
    }

    return stats


def mcf(
    target_spectrum,
    library,
    assignment_radius=np.inf,
    absorption_cost=None,
    dist_pars={},
    target_id=None,
    lib_id=None,
    savefn=None,
    target_regions=None,
    isolated_fit=False,
    resolveYinResult=False,
    load=True,
    load_dists=True,
):
    """
    Minimum cost flow assignment
    target_spectrum:
        structure with attribute 'coords' and 'weights' describing
        the target spectrum as a list of weighted points
    library:
        dict mapping compound IDs to compound spectra (provided in
        the same form as target_spectrum)
    assignment_radius:
        maximal distance between two peaks for matching them
    absorption_cost:
        specific cost of absorbed flow, default=config.MAX_ABSORPTON_COST
    dist_pars:
        parameter dict for the distance to be used.
        Currently, the only implemented distance is euclidean (2-norm)
        with weighted x- and y-directions. These weights are controlled
        by 'scalex' and 'scaley' with resulting distance

            dist(x,y) = sqrt((x[0]-y[0])**2/scalex**2 + (x[1]-y[1])**2/scaley**2)

        per default scalex=10.0 and scaley=1.0, reflecting scalings for
        C- and H-direction in HSQC spectra.
    target_id:
        allows to specify an ID for the target spectrum. If target_id and lib_id
        are given, the result is saved automatically under

            OUTDIR/<target_id>_<lib_id>_ar<assignment_radius>.pickle

    savefn:
        Allows to provide a specific filepath to save the result. (expects a pathlib.Path)
    isolated_fit:
        whether all compounds should be fitted independently. Otherwise a joint fit
        is executed, which avoids overassignment of flow to target nodes.
    resolveYinResult:
        whether to include inflows and residuals for nodes in target spectrum
        (needs a considerable amount of memory for raster targets, but is required
        for incremental fitting, default: False)
    load:
        If true, the cached result is returned if it exists.
    load_dists:
        If true, the cached dist-lists are used for the weights of peak node connections.
    """
    cache_dir = OUTDIR / "cache" / "mcf"
    if not cache_dir.exists():
        os.makedirs(cache_dir)
        print(f"Created directory '{cache_dir}'")
    if savefn is None and target_id is not None and lib_id is not None:
        savefn = cache_dir / (
            f"mcfResult_{target_id}_by_{lib_id}_ar%g.pickle" % assignment_radius
        )
    if load and savefn is not None and savefn.exists():
        # Load result
        with open(savefn, "rb") as f:
            res = pickle.load(f)
        print(f"Loaded MCF result from '{savefn}'")
        return res

    if absorption_cost is None:
        absorption_cost = MAX_ABSORPTON_COST

    _, weightsY, coordsY = serialize(dict(target_spectrum=target_spectrum))

    compound_ix, weightsX, coordsX = serialize(library)
    target_regions = load_regions(target_regions)

    scalex = dist_pars.get("scalex", DEFAULT_SCALE_C)
    scaley = dist_pars.get("scaley", DEFAULT_SCALE_H)
    scales = np.array([scalex, scaley])

    # Collect timing infos
    timings = {}

    tic = default_timer()
    print("Making dist list for ar=%g ..." % assignment_radius)
    distlist_fn = cache_dir / (
        f"dist_list_{target_id}_{lib_id}_ar%g.pickle" % assignment_radius
    )
    if load_dists and distlist_fn.exists():
        with open(distlist_fn, "rb") as f:
            dist_info = pickle.load(f)
        print(f"Loaded distlists from '{distlist_fn}'")
        computed_dists = False
    else:
        dist_info = euclidean_dists(coordsX, coordsY, scales, assignment_radius)
        with open(distlist_fn, "wb") as f:
            pickle.dump(dist_info, f)
        print(f"Saved distlists to '{distlist_fn}'")
        computed_dists = True
    timings["distList"] = default_timer() - tic
    print("makeDistList() took %g secs" % timings["distList"])

    # Determine which points to exclude based on regions
    print("Checking containment...")
    tic = default_timer()

    ixXinY = getIndicesWithin(
        target_regions,
        coordsX,
        target_id=f"lib {lib_id} in {target_id}",
        cache_dir=cache_dir,
        load=load_dists and not (computed_dists),
    )
    ixYinY = getIndicesWithin(
        target_regions,
        coordsY,
        target_id,
        cache_dir=cache_dir,
        load=load_dists and not (computed_dists),
    )
    timings["containmentCheck"] = default_timer() - tic
    print("Containment checking took %g secs" % timings["containmentCheck"])

    # Generate flow graph
    print("Generating flow net...")
    tic = default_timer()
    flownet = makeMCFNet(
        dist_info,
        weightsY,
        compound_ix,
        weightsX,
        absorption_cost,
        isolated_fit=isolated_fit,
        ixYinY=ixYinY,
        ixXinY=ixXinY,
    )
    toc = default_timer()
    print("Generated flowNet:\n", flownet)
    timings["flowNetGeneration"] = toc - tic
    print("Generation took %g secs" % timings["flowNetGeneration"])

    # Calculate min-cost-flow
    print("Calculating min cost flow...")
    tic = default_timer()
    calculateMCF(flownet)
    toc = default_timer()
    timings["minCostFlow"] = toc - tic
    print("Min-cost-flow calculation took %g secs" % timings["minCostFlow"])

    # Make stats
    print("Collecting stats...")
    tic = default_timer()
    stats = flowstats(flownet, resolveYinResult)
    total_target_weight = sum(weightsY[ixYinY])
    result = MCFResult(
        total_target_weight, stats, absorption_cost, flownet.graph["compounds"], timings
    )
    timings["statsCollection"] = default_timer() - tic
    print("Stats collection took %g secs\n" % timings["statsCollection"])

    if DEBUG > 0:
        print("\nTimings (at nX=%s, nY=%d):" % (len(coordsX), len(weightsY)))
        for k, v in timings.items():
            print("%s: %g secs" % (k, v))

    # Save results
    if savefn is not None:
        with open(savefn, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved MCF result to '{savefn}'")

    return result
