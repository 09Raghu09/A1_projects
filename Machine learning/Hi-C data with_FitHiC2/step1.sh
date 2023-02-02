# Step 1 of protocol
# generate contact maps file from valid pairs data

FITHICDIR=myvenv/lib/python3.6/site-packages/fithic
DATADIR=data

echo -e "######\nStarting Step 1.\n######"

bash $FITHICDIR/utils/validPairs2FitHiC-fixedSize.sh 40000 IMR90 $DATADIR/validPairs/IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz $DATADIR/contactCounts
