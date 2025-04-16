import pandas as pd

from mcfnmr.demodata.loading import loadOLDBCompoundLib, loadOLDB_1D_test 
from mcfnmr.core.mcf import mcf
from mcfnmr.routines.classification import classifyCompounds
from mcfnmr.demodata import mix4
from pprint import pp
from mcfnmr.main import classify_result, save_as_text


def main():
    # load 2D-spectrum inhouse lib as 1D-projection (just for testing!),
    # and include water, DMSO, and TMSP
    lib = loadOLDBCompoundLib(project_1D = True, add_extra = True)
    
    # load 1D spectrum
    # 204 == IIa ("Mixture: D-Glucose, Pimelic acid, 1,2,3-Tricarboxylic acid, Tyrosine (all 30 mM)")
    targetID = "nmr_IIa_1H"
    containedCompounds = mix4 + ["DMSO", "Water", "TMSP"]
    detection_threshold = 1e14
    
    # targetID = "204_1H_solvent_supression"
    # containedCompounds = mix4 + ["DMSO", "Water", "TMSP"]
    # detection_threshold = 1e14
        
    target = loadOLDB_1D_test(targetID, binsize=10)
    
    # # Debug: Plot
    # import matplotlib.pyplot as plt
    # plt.plot(target.coords[:,1], target.weights)
    # import numpy as np
    # plt.plot(target.coords[:,1], np.minimum(0.0, target.weights), color="r")
    # plt.show()
    
    # run recombination
    lib_id = "inhouse_extended"
    res = mcf(target_spectrum=target, library=lib, 
              target_id = target.name, lib_id = lib_id,
              assignment_radius=0.1,
            load=True,
            load_dists=True,)
    
    
    df = classify_result(res, detection_threshold)
    save_as_text(df, "output/mcfresult_test.csv")
    
    nC = len(res.compounds)
    compounds = sorted(res.compounds)
    
    pars = dict(libID=lib_id, targetSpecID=targetID)
    dfCompounds = dict(
        lib=[lib_id]*nC,
        target=[targetID]*nC,
        compound=compounds,
        assigned=res.assigned,
        nPeaks=res.nPeaksConsidered,
        )
    dfCompounds = pd.DataFrame(dfCompounds)
    
    classification = classifyCompounds(dfCompounds, pars, th=detection_threshold, 
                      containedCompounds=containedCompounds, 
                      scale_by_peaknr=False, 
                      verb=False)
    
    del classification["compound_df"]
    print("Result:")
    pp(classification)
    
    
if __name__=="__main__":
    main()


