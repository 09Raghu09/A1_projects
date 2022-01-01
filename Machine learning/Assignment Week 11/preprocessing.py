from pyopenms import *
from pathlib import Path
import sys

def smoothe(inpath, outdir):
    exp = MSExperiment()
    gf = GaussFilter()
    param = gf.getParameters()
    gaussian_bandwidths = [1.2, 1.3, 1.5]

    for width in gaussian_bandwidths:
        param.setValue("gaussian_width", width)
        gf.setParameters(param)

        fh = MzMLFile()
        fh.load(str(inpath), exp)
        gf.filterExperiment(exp)
        outname = str(inpath.stem)+"gaus"+str(width)+".mzML"
        outpath = outdir / outname
        fh.store(str(outpath), exp)

def main(inpaths, outdir):
    for inpath in inpaths:
        smoothe(inpath, outdir)



if __name__ == '__main__':
    data_dirpath = Path("output/features")
    class0_filename = "menRCCstage2_981.featureXML"
    class1_filename = "womenLCCstage2_95.featureXML"

    class00_filename = "menLCCstage3_988.featureXML"
    class11_filename = "men_normal_6.featureXML"

    # input_dir = data_dirpath / "input"
    # output_dir = data_dirpath / "output"
    input_dir = data_dirpath
    output_dir = Path("output") / "preprocessed"

    inpaths = [data_dirpath/class0_filename, data_dirpath/class1_filename]
               #data_dirpath/class00_filename, data_dirpath/class11_filename]

    main(inpaths, output_dir)
