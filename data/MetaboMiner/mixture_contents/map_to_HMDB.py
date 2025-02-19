import editdistance
from mcfnmr.config import MM_MIXLISTSDIR
from mcfnmr.demodata.loading import userprovided_HMDB_IDs, buildName2HMDBIDMap


def lookup_most_similar(name, names):
    minDist = 1e12
    best_match = None
    for i, c in enumerate(names):
        dist = editdistance.eval(name, c)
        if dist < minDist:
            best_match = c
            minDist = dist
    return best_match, minDist


def findHMDB_ID(name, map_name2HMDB, all_names_cached=None):
    if name in userprovided_HMDB_IDs:
        HMDB_ID = userprovided_HMDB_IDs[name]
        print(f"{name} → {HMDB_ID} (user-provided)")
        if HMDB_ID is None:
            print(f"   → ignoring {name}!")
    elif name in map_name2HMDB:
        HMDB_ID = map_name2HMDB[name]
        print(f"{name} → {HMDB_ID}")
    else:
        if all_names_cached is None:
            all_names_cached = sorted(map_name2HMDB.keys())
        best_match, dist = lookup_most_similar(name, all_names_cached)
        HMDB_ID = map_name2HMDB[best_match]
        print(f"{name} → {best_match} → {HMDB_ID}")
        print("  (dist = %d)" % dist)
    return str(HMDB_ID)


def convertMixListsToHMDB():
    map_name2HMDB = buildName2HMDBIDMap()
    all_names = sorted(map_name2HMDB.keys())

    print("Converting names to HMDB-IDs in mixture information.")
    for mixID in ["N880", "N907", "N925", "N926", "N987", "N988"]:
        fn_in = MM_MIXLISTSDIR / (mixID + ".txt")
        fn_out = MM_MIXLISTSDIR / (mixID + "_HMDB.txt")

        with open(fn_in, "r") as fin:
            ll = fin.readlines()
        with open(fn_out, "w") as fout:
            for l in ll:
                name = l.strip()
                if name == "":
                    # empty line
                    continue
                HMDB_ID = findHMDB_ID(name, map_name2HMDB, all_names)
                if HMDB_ID is None:
                    continue
                fout.write(HMDB_ID + "\n")


if __name__ == "__main__":
    convertMixListsToHMDB()
