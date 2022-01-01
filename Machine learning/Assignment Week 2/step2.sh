# Step 2 of protocol
# generate fragment mappability file 

FITHICDIR=myvenv/lib/python3.6/site-packages/fithic
DATADIR=data

echo -e "######\nStarting Step 2.\n######"

python3 $FITHICDIR/utils/createFitHiCFragments-fixedsize.py --chrLens $DATADIR/referenceGenomes/hg19wY-lengths --resolution 40000 --outFile $DATADIR/fragmentMappability/IMR90_fithic.fragmentsfile.gz
