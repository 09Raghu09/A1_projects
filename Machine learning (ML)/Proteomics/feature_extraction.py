from urllib.request import urlretrieve
from pathlib import Path
from pyopenms import *
import sys
import numpy as np
import pandas as pd
import os

def extract_features(inpath, outdir):
    instem = inpath.stem
    print("Starting feature extraction of " + str(instem) + ".")
    options = PeakFileOptions()
    options.setMSLevels([1])
    fh = MzMLFile()
    fh.setOptions(options)

    # Load data
    input_map = MSExperiment()

    print("Inpath", inpath)
    fh.load(str(inpath), input_map)
    input_map.updateRanges()

    ff = FeatureFinder()
    ff.setLogType(LogType.CMD)

    # Run the feature finder
    name = "centroided"
    features = FeatureMap()
    seeds = FeatureMap()
    params = FeatureFinder().getParameters(name)
    ff.run(name, input_map, features, params, seeds)

    features.setUniqueIds()
    fh = FeatureXMLFile()
    outname = instem + ".featureXML"
    outpath = str(outdir/outname)
    print("Outpath:", outpath)
    # fh.store("output.featureXML", features)
    fh.store(outpath, features)
    print("Found", features.size(), "features")
    return [[f.getRT(), f.getMZ()] for f in features]


def main(inpaths, outdir):
    tidy = pd.DataFrame()
    for index, inpath in enumerate(inpaths):
        feats = extract_features(inpath, outdir)
        for feat in feats: feat.append(index)
        tidy.append(feats)

    tidy.to_csv(str(outdir/"all_feats_classified.csv"), sep="\t")



if __name__ == '__main__':
    data_dirpath = Path("MTBLS1129")
    class0_filename = "menRCCstage2_981.mzML"
    class1_filename = "womenLCCstage2_95.mzML"

    # class00_filename = "menLCCstage3_988.mzML"
    # class11_filename = "men_normal_6.mzML"


    input_dir = data_dirpath
    outdir = Path("output") / "features"

    if len(sys.argv) > 1:
        inpaths = sys.argv[0:]
    else:
        inpaths = [data_dirpath/class0_filename, data_dirpath/class1_filename]


    if not os.path.exists(outdir):
        outdir.mkdir(parents=True, exist_ok=True)

    main(inpaths, outdir)
