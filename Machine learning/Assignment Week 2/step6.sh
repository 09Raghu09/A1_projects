# Step 6 of protocol
# run fithic

FITHICDIR=myvenv/lib/python3.6/site-packages/fithic
DATADIR=data

echo -e "######\nStarting Step 6.\n######"

python3 $FITHICDIR/fithic.py -i $DATADIR/contactCounts/Rao_GM12878-primary-chr5_5kb.gz -f $DATADIR/fragmentMappability/Rao_GM12878-primary-chr5_5kb.gz -t $DATADIR/biasValues/Rao_GM12878-primary-chr5_5kb.gz -r 5000 -o $DATADIR/fithicOutput/Rao_GM12878-primary-chr5_5kb -l Rao_GM12878-primary-chr5_5kb -U 1000000 -L 15000 -v
