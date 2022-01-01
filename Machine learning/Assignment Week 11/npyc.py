from pathlib import Path
import nPYc

inpath = Path("stage2_woman_men_feature_tidy.csv")
inpath2 = Path("output/features/womenLCCstage2_95.featureXML")
inpath3 = Path("output/preprocessed/womenLCCstage2_95gaus1.3.mzML")

paths=[inpath, inpath2, inpath3]

for path in paths:
    try:
        msData = nPYc.MSDataset(path)
        print(msData)
    except :
        print("Failed "+ str(path))

print("-------------------------------------")

#for path in paths:
for path in [inpath2]:
    msData = nPYc.MSDataset(path)
    print(msData)
