from copy import deepcopy


# Different libraries provided by MM
metaboMinerLibNames = [
    "Biofluid ( all )",
    "Biofluid ( common )",
    "CSF ( all )",
    "CSF ( common )",
    "Plasma ( all )",
    "Plasma ( common )",
    "Urine ( all )",
    "Urine ( common )",
]
metaboMinerLibNameDict = {
    "all" : "Biofluid ( all )",
    "common": "Biofluid ( common )",
    "csf-all": "CSF ( all )",
    "csf-common": "CSF ( common )",
    "plasma-all": "Plasma ( all )",
    "plasma-common": "Plasma ( common )",
    "urine-all": "Urine ( all )",
    "urine-common": "Urine ( common )",
}

# Subsets of MaetaboMiner mixes
mix_selections = {
    "ph7": ["N925", "N926", "N987", "N988"],
    "SMART-Miner": ["N925", "N987", "N988"],
    "allMM": ["N880", "N907", "N925", "N926", "N987", "N988"],
}

# Bin sizes for raster data
binning = {
    "N907": (1, 4),
    "N925": (2, 4),
    "N987": (1, 2),
    "N926": (1, 2),
    "N880": (1, 2),
    "N988": (1, 2),
}


# Template & defaults
caseSpec0 = dict(
    name="default",
    # not using peak weights for comparison as this is not used by metabominer, either
    uniformPeaks=True,
    # Whether to use the point spectrum or the raster spectrum as target
    pointTarget=True,
    # Whether to fit compounds individually
    isolated_fit=False,
    # Should this fit build on previous ones (with lower assingment radius)?
    incrementalFit=False,
    # MetaboMiner Compound lib
    libID="metabominer",
    mixIDs=["N925", "N987", "N988"],  # SMART-Miner mixes
)

caseSpec1 = deepcopy(caseSpec0)
caseSpec1["name"] = "base case"

## Setups for Paper
# A: joint, single pass, grid
caseSpecA = deepcopy(caseSpec1)
caseSpecA["pointTarget"] = False
caseSpecA["name"] = "setup A"

# B: independent, single pass, grid
caseSpecB = deepcopy(caseSpecA)
caseSpecB["isolated_fit"] = True
caseSpecB["name"] = "setup B"

# C: joint, incremental, grid
caseSpecC = deepcopy(caseSpecA)
caseSpecC["incrementalFit"] = True
caseSpecC["name"] = "setup C"

# D: independent, single pass, peak list
caseSpecD = deepcopy(caseSpecA)
caseSpecD["pointTarget"] = True
caseSpecD["name"] = "setup D"

# E: joint, incremental, peak list
caseSpecE = deepcopy(caseSpecA)
caseSpecE["pointTarget"] = True
caseSpecE["incrementalFit"] = True
caseSpecE["name"] = "setup E"

# E: joint, incremental, peak list
caseSpecF = deepcopy(caseSpecA)
caseSpecF["pointTarget"] = True
caseSpecF["isolated_fit"] = True
caseSpecF["name"] = "setup F"

baseSpec = dict(
    A=caseSpecA, B=caseSpecB, C=caseSpecC, D=caseSpecD, E=caseSpecE, F=caseSpecF
)


# For individual mixes and setups from paper
# map setup -> mix -> pars
case_dict = dict(A={}, B={}, C={}, D={}, E={}, F={})
for setup in ["A", "B", "C", "D", "E", "F"]:
    for mix in ["N907", "N925", "N926", "N987", "N988", "N880", "all"]:
        mix_list = ["N925", "N987", "N988"] if mix == "all" else [mix]
        case_dict[setup][mix] = deepcopy(baseSpec[setup])
        case_dict[setup][mix]["mixIDs"] = mix_list

## Make case dicts for A, B, C for quantification single runs (no multiple mixes allowed)
for series in ["I", "II", "III"]:
    for experiment in ["a", "b", "c"]:
        for setup in ["A", "B", "C"]:
            mixID = series + "." + experiment
            case_dict[setup][mixID] = deepcopy(baseSpec[setup])
            case_dict[setup][mixID]["mixIDs"] = [mixID]

# Map compound full names of  OLDB-lib to abbreviations
compoundShortNames = {
    "Raffinose": "Raf",
    "Pimelic acid": "Pim",
    "D-Xylose": "Xyl",
    "L-Cysteine": "Cys",
    "Nicotinic acid": "Nic",
    "Malonic acid": "Mal",
    "L-Proline": "Pro",
    "Anthranilic acid": "Ant",
    "Citric acid": "Cit",
    "L-Rhamnose": "Rha",
    "1,2,4-Benzenetricarboxylic acid": "Bnz4",
    "1,2,3-Benzenetricarboxylic acid": "Bnz3",
    "D-Mannose": "Man",
    "L-Glutamine": "Glu",
    "4-Phenylbutanoic acid": "PheB",
    "Succinic acid": "Suc",
    "Benzoic acid": "Bnz",
    "Octanoic acid": "Oct",
    "Biotin": "Bio",
    "4-Hydroxybenzoic acid": "HBz",
    "9-Anthracenecarboxylic acid": "AntCx",
    "L-Isoleucine": "ILeu",
    "L-Tartaric acid": "Tar",
    "L-Tyrosine": "Tyr",
    "L-Tryptophane": "Try",
    "L-Glutamic acid": "GluA",
    "L-Methionine": "Met",
    "L-Phenylalanine": "PheA",
    "4-Hydroxycinnamic acid": "HCin",
    "D-Glucose": "Gluc",
    "L-Leucine": "LLeu",
    "(E)-Cinnamic acid": "CinA",
    "Tetradecanoic acid": "TDec",
    "Vanillic acid": "Van",
}

# Inverse map
compoundLongNames = dict([(v, k) for k, v in compoundShortNames.items()])

# Updated short names
textbook_abbreviations = {
    "Raffinose": "Raf",
    "Pimelic acid": "PA",
    "D-Xylose": "Xyl",
    "L-Cysteine": "Cys",
    "Nicotinic acid": "NA",
    "Malonic acid": "MA",
    "L-Proline": "Pro",
    "Anthranilic acid": "AA",
    "Citric acid": "CA",
    "L-Rhamnose": "Rha",
    "1,2,4-Benzenetricarboxylic acid": "1,2,4-BTCA",
    "1,2,3-Benzenetricarboxylic acid": "1,2,3-BTCA",
    "D-Mannose": "Man",
    "L-Glutamine": "Gln",
    "4-Phenylbutanoic acid": "4-PBA",
    "Succinic acid": "SA",
    "Benzoic acid": "BA",
    "Octanoic acid": "OA",
    "Biotin": "Biotin",
    "4-Hydroxybenzoic acid": "4-HBA",
    "9-Anthracenecarboxylic acid": "9-ACA",
    "L-Isoleucine": "Ile",
    "L-Tartaric acid": "TA",
    "L-Tyrosine": "Tyr",
    "L-Tryptophane": "Trp",
    "L-Glutamic acid": "Glu",
    "L-Methionine": "Met",
    "L-Phenylalanine": "Phe",
    "4-Hydroxycinnamic acid": "4-HCA",
    "D-Glucose": "Glc",
    "L-Leucine": "Leu",
    "(E)-Cinnamic acid": "tCA",
    "Tetradecanoic acid": "TDA",
    "Vanillic acid": "VA",
}

# Some substances have aliases
correctSubstrateNames = {
    "1,2,3-Benzentricarboxylic acid": "1,2,3-Benzenetricarboxylic acid",
    "1,2,4-Benzentricarboxylic acid": "1,2,4-Benzenetricarboxylic acid",
}


def correctShortCompoundLabels(compound_list):
    res = []
    for c in compound_list:
        clong = compoundLongNames.get(c, c)
        cnew = textbook_abbreviations[clong]
        res.append(cnew)
    return res


# Map mix identifiers to descriptive names
mixNamesOLDB = {}
for s1 in ["I", "II", "III"]:
    for s2 in ["a", "b", "c"]:
        mixNamesOLDB[s1 + s2 + "_01"] = s1 + s2 + "_01"
        mixNamesOLDB[s1 + s2] = s1 + s2

# Number of NMR scans by which the OLDB spectra were recorded
# Intensity scales linearly with the scan number (used for
# renormalization to predict the concentration)
scanNumberIndividualCompounds = 4
scanNumbersMixes = {
    "Ia_01": 8,
    "Ia_02": 8,
    "Ia_03": 16,
    "Ia_04": 16,
    "Ib_01": 16,
    "Ic_01": 64,
    "IIa_01": 4,
    "IIa_03": 4,
    "IIa_04": 4,
    "IIb_01": 8,
    "IIc_01": 16,
    "IIIa_01": 8,
    "IIIb_01": 64,
    "IIIc_01": 32,
}


intensityScalings = {
    "(E)-Cinnamic acid": 3,
    "1,2,3-Benzenetricarboxylic acid": 2,
    "1,2,4-Benzenetricarboxylic acid": 1,
    "4-Hydroxybenzoic acid": 3,
    "4-Hydroxycinnamic acid": 3,
    "4-Phenylbutanoic acid": 3,
    "9-Anthracenecarboxylic acid": 3,
    "Anthranilic acid": 2,
    "Benzoic acid": 2,
    "Biotin": 3,
    "Citric acid": 2,
    "D-Glucose": 2,
    "D-Mannose": 3,
    "D-Xylose": 2,
    "L-Cysteine": 2,
    "L-Glutamic acid": 2,
    "L-Glutamine": 2,
    "L-Isoleucine": 3,
    "L-Leucine": 4,
    "L-Methionine": 3,
    "L-Phenylalanine": 3,
    "L-Proline": 2,
    "L-Rhamnose": 3,
    "L-Tartaric acid": 4,
    "L-Tryptophane": 3,
    "L-Tyrosine": 3,
    "Malonic acid": 3,
    "Nicotinic acid": 2,
    "Octanoic acid": 4,
    "Pimelic acid": 4,
    "Raffinose": 4,
    "Succinic acid": 3,
    "Tetradecanoic acid": 6,
    "Vanillic acid": 4,
    # mixtures
    "Ia_01": 7,
    "Ia_02": 7,
    "Ia_03": 8,
    "Ia_04": 7,
    "Ib_01": 4,
    "Ic_01": 5,
    "IIa_01": 7,
    "IIa_03": 7,
    "IIa_04": 7,
    "IIb_01": 5,
    "IIc_01": 3,
    "IIIa_01": 7,
    "IIIb_01": 7,
    "IIIc_01": 5,
}


# Total intensity weights computed from single compound raster
# spectra in "exp" at noiseThreshold = 5*noiseStd
OLDBCompoundWeights = {
    "(E)-Cinnamic acid": 21859012.496749796,
    "1,2,3-Benzenetricarboxylic acid": 22211186.877651516,
    "1,2,4-Benzenetricarboxylic acid": 28527385.839007806,
    "4-Hydroxybenzoic acid": 17132620.513385423,
    "4-Hydroxycinnamic acid": 22403039.00800274,
    "4-Phenylbutanoic acid": 36354978.7523742,
    "9-Anthracenecarboxylic acid": 35001172.65861179,
    "Anthranilic acid": 36086178.92969638,
    "Benzoic acid": 29635603.682999723,
    "Biotin": 56740359.20936215,
    "Citric acid": 36143796.919918396,
    "D-Glucose": 97412798.24552464,
    "D-Mannose": 48325549.56628264,
    "D-Xylose": 81751109.9075893,
    "L-Cysteine": 26482232.482687145,
    "L-Glutamic acid": 37998891.80201849,
    "L-Glutamine": 44890563.45445999,
    "L-Isoleucine": 33853864.25405648,
    "L-Leucine": 18690289.75304223,
    "L-Methionine": 26495591.527344096,
    "L-Phenylalanine": 35685236.67435031,
    "L-Proline": 52887570.22402762,
    "L-Rhamnose": 47923534.29768606,
    "L-Tartaric acid": 6971518.753678279,
    "L-Tryptophane": 39452109.81456034,
    "L-Tyrosine": 35627413.78334053,
    "Malonic acid": 9633356.927163415,
    "Nicotinic acid": 19466861.819191758,
    "Octanoic acid": 22403034.708915677,
    "Pimelic acid": 20947561.12618703,
    "Raffinose": 63519645.654627636,
    "Succinic acid": 10103609.511354258,
    "Tetradecanoic acid": 10857794.299946425,
    "Vanillic acid": 15317233.912271747,
}

# Individual compound spectra were recorded at 3 mM. This yields expected concentration factors
# for each compound in the recombination dependent on the mix. These are recorded here.
mix4 = ["D-Glucose", "Pimelic acid", "1,2,3-Benzenetricarboxylic acid", "L-Tyrosine"]
expectedConcentrationFactors = {
    "Ia_01": dict([(c, 1.0) for c in OLDBCompoundWeights]),
    "Ib_01": dict([(c, 0.1) for c in OLDBCompoundWeights]),
    "Ic_01": dict([(c, 0.01) for c in OLDBCompoundWeights]),
    "IIa_01": dict([(c, 10.0) if c in mix4 else (c, 0.0) for c in OLDBCompoundWeights]),
    "IIb_01": dict([(c, 1.0) if c in mix4 else (c, 0.0) for c in OLDBCompoundWeights]),
    "IIc_01": dict([(c, 0.1) if c in mix4 else (c, 0.0) for c in OLDBCompoundWeights]),
    "IIIa_01": dict(
        [(c, 5.05) if c in mix4 else (c, 0.05) for c in OLDBCompoundWeights]
    ),
    "IIIb_01": dict(
        [(c, 0.55) if c in mix4 else (c, 0.05) for c in OLDBCompoundWeights]
    ),
    "IIIc_01": dict([(c, 0.1) if c in mix4 else (c, 0.05) for c in OLDBCompoundWeights]),
}



