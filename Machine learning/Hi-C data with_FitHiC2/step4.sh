# Step 4 of protocol
# computes bias values

FITHICDIR=myvenv/lib/python3.6/site-packages/fithic
DATADIR=data

echo -e "######\nStarting Step 4.\n######"

[ ! -d $DATADIR/biasValues ] && mkdir $DATADIR/biasValues

python3 $FITHICDIR/utils/HiCKRy.py -i $DATADIR/contactCounts/Duan_yeast_EcoRI.gz -f $DATADIR/fragmentMappability/Duan_yeast_EcoRI.gz -o $DATADIR/biasValues/Duan_yeast_EcoRI.gz

python3 $FITHICDIR/utils/HiCKRy.py -i $DATADIR/contactCounts/Dixon_IMR90-wholegen_40kb.gz -f $DATADIR/fragmentMappability/Dixon_IMR90-wholegen_40kb.gz -o $DATADIR/biasValues/Dixon_IMR90-wholegen_40kb.gz

python3 $FITHICDIR/utils/HiCKRy.py -i $DATADIR/contactCounts/Rao_GM12878-primary-chr5_5kb.gz -f $DATADIR/fragmentMappability/Rao_GM12878-primary-chr5_5kb.gz -o $DATADIR/biasValues/Rao_GM12878-primary-chr5_5kb.gz
