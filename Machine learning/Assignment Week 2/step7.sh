# Step 7 of protocol
# depict fithic result as html

FITHICDIR=myvenv/lib/python3.6/site-packages/fithic
DATADIR=data

echo -e "######\nStarting Step 7.\n######"

bash $FITHICDIR/utils/createFitHiCHTMLout.sh Rao_GM12878-primary-chr5_5kb 1 $DATADIR/fithicOutput/Rao_GM12878-primary-chr5_5kb
