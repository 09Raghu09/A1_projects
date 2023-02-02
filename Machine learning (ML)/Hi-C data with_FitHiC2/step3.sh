# Step 3 of protocol
# generate fragment mappability file for non-fixed-size dataset. Run digestions simulation by HindIII deoxyribonuclease restriction enzyme

FITHICDIR=myvenv/lib/python3.6/site-packages/fithic
DATADIR=data

echo -e "######\nStarting Step 3.\n######"

bash $FITHICDIR/utils/createFitHiCFragments-nonfixedsize.sh $DATADIR/fragmentMappability/yeast_fithic.fragments HindIII $DATADIR/referenceGenomes/yeast_reference_sequence_R62-1-1_20090218.fsa
