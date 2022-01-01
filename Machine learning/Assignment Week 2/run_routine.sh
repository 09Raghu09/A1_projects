# Team N
# runs protocoll steps 1-4 according to protocol “Identifying statistically significant chromatin contacts from Hi-C data with FitHiC2” by Arya Kaul et al. (https://doi.org/10.1038/s41596-019-0273-0).

FITHICDIR=myvenv/lib/python3.6/site-packages/fithic
DATADIR=data

./step0.sh
[ ! -f $DATADIR/contactCounts/IMR90_fithic.contactCounts.gz ] && ./step1.sh  # only do that if needed, takes up to 1 hour
./step2.sh
./step3.sh
./step4.sh
./step6.sh
./step7.sh
