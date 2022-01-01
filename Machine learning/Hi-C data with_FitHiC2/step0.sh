# makes sure the software environment is all set up and needed input data is provided

FITHICDIR=myvenv/lib/python3.6/site-packages/fithic
DATADIR=data

echo -e "######\nStarting Step 0.\n######"

# python virtualenv must be named 'myvenv', gets created otherwise
[ ! -d "myvenv" ] && echo -e "Creating python virtual environment 'myvenv'" && python3.6 -m venv myvenv
echo "Activating venv" 
source myvenv/bin/activate
sleep 1

echo -e "Checking if latest fithic version and all dependencies are installed via pip" && pip install fithic

# check for and download missing fithic utils
[ ! -d $FITHICDIR/utils ] && mkdir $FITHICDIR/utils
[ -f $FITHICDIR/utils/validPairs2FitHiC-fixedSize.sh ] && echo "Util 'validPairs2FitHiC-fixedSize.sh' found in '$FITHICDIR/utils/'"
[ ! -f $FITHICDIR/utils/validPairs2FitHiC-fixedSize.sh ] && echo -e "Downloading fithic util from: \n https://raw.githubusercontent.com/ay-lab/fithic/master/fithic/utils/validPairs2FitHiC-fixedSize.sh" && wget https://raw.githubusercontent.com/ay-lab/fithic/master/fithic/utils/validPairs2FitHiC-fixedSize.sh && mv validPairs2FitHiC-fixedSize.sh $FITHICDIR/utils

[ -f $FITHICDIR/utils/createFitHiCFragments-nonfixedsize.sh ] && echo "Util 'createFitHiCFragments-nonfixedsize.sh' found in '$FITHICDIR/utils/'"
[ ! -f $FITHICDIR/utils/createFitHiCFragments-nonfixedsize.sh ] && echo -e "Downloading fithic util createFitHiCFragments-nonfixedsize.sh from: \n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/code/fithic/fithic/utils/createFitHiCFragments-nonfixedsize.sh && mv createFitHiCFragments-nonfixedsize.sh $FITHICDIR/utils

[ -f $FITHICDIR/utils/createFitHiCFragments-fixedsize.py ] && echo "Util 'createFitHiCFragments-fixedsize.py' found in '$FITHICDIR/utils/'"
[ ! -f $FITHICDIR/utils/createFitHiCFragments-fixedsize.py ] && echo -e "Downloading fithic util createFitHiCFragments-fixedsize.py from: \n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/code/fithic/fithic/utils/createFitHiCFragments-fixedsize.py && mv createFitHiCFragments-fixedsize.py $FITHICDIR/utils

[ -f $FITHICDIR/utils/HiCKRy.py ] && echo "Util 'HiCKRy.py' found in '$FITHICDIR/utils/'"
[ ! -f $FITHICDIR/utils/HiCKRy.py ] && echo -e "Downloading fithic util HiCKRy.py from: \n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/code/fithic/fithic/utils/HiCKRy.py && mv HiCKRy.py $FITHICDIR/utils

[ -f $FITHICDIR/utils/createFitHiCHTMLout.sh ] && echo "Util 'createFitHiCHTMLout.sh' found in '$FITHICDIR/utils/'"
[ ! -f $FITHICDIR/utils/createFitHiCHTMLout.sh ] && echo -e "Downloading fithic util createFitHiCHTMLout.sh from: \n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/code/fithic/fithic/utils/createFitHiCHTMLout.sh 
[ -f createFitHiCHTMLout.sh ] && mv createFitHiCHTMLout.sh $FITHICDIR/utils

# check for and download missing contact counts
[ ! -d $DATADIR/contactCounts ] && mkdir $DATADIR/contactCounts
[ -d $DATADIR/contactCounts ] && echo -e "Contact counts data found in '$DATADIR/contactCounts/'"
[ ! -f $DATADIR/contactCounts/Dixon_IMR90-wholegen_40kb.gz ] && echo -e "Downloading contact counts file Dixon_IMR90-wholegen_40kb.gz  from:\n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/contactCounts/Dixon_IMR90-wholegen_40kb.gz && mv Dixon* $DATADIR/contactCounts/
[ ! -f $DATADIR/contactCounts/Duan_yeast_EcoRI.gz ] && echo -e "Downloading contact counts file Duan_yeast_EcoRI.gz from:\n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/contactCounts/Duan_yeast_EcoRI.gz && mv Duan* $DATADIR/contactCounts/
[ ! -f $DATADIR/contactCounts/Rao_GM12878-primary-chr5_5kb.gz ] && echo -e "Downloading contact counts file Rao_GM12878-primary-chr5_5kb.gz  from:\n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/contactCounts/Rao_GM12878-primary-chr5_5kb.gz && mv Rao* $DATADIR/contactCounts/

# check for and download missing reference genomes
[ ! -d $DATADIR/referenceGenomes ] && mkdir $DATADIR/referenceGenomes
[ -d $DATADIR/referenceGenomes ] && echo -e "Reference genome dir found: '$DATADIR/referenceGenomes /'"
[ ! -f $DATADIR/referenceGenomes/hg19wY-lengths ] && echo -e "Downloading reference genome hg19wY-lengths from:\n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/referenceGenomes/hg19wY-lengths && mv hg19wY* $DATADIR/referenceGenomes
[ ! -f $DATADIR/referenceGenomes/yeast_reference_sequence_R62-1-1_20090218.fsa ] && echo -e "Downloading reference genome yeast_reference_sequence_R62-1-1_20090218.fsa from:\n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/referenceGenomes/yeast_reference_sequence_R62-1-1_20090218.fsa && mv yeast_reference_* $DATADIR/referenceGenomes


# check for and download missing fragment mappability data
[ ! -d $DATADIR/fragmentMappability ] && mkdir $DATADIR/fragmentMappability
[ ! -f $DATADIR/fragmentMappability/Rao_GM12878-primary-chr5_5kb.gz ] && echo -e "Downloading fragment mappability data from:\n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/fragmentMappability/Rao_GM12878-primary-chr5_5kb.gz && mv Rao_GM12878-primary-chr5_5kb.gz $DATADIR/fragmentMappability
[ ! -f $DATADIR/fragmentMappability/Duan_yeast_EcoRI.gz ] && echo -e "Downloading fragment mappability data from:\n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/fragmentMappability/Duan_yeast_EcoRI.gz && mv Duan_yeast_EcoRI.gz $DATADIR/fragmentMappability
[ ! -f $DATADIR/fragmentMappability/Dixon_IMR90-wholegen_40kb.gz ] && echo -e "Downloading fragment mappability data from:\n https://files.codeocean.com/" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/fragmentMappability/Dixon_IMR90-wholegen_40kb.gz && mv Dixon_IMR90-wholegen_40kb.gz $DATADIR/fragmentMappability

# check for and download missing input dataset
[ ! -d $DATADIR/validPairs ] && mkdir $DATADIR/validPairs
[ -f $DATADIR/validPairs/IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz ] && echo -e "Input pair data found in '$DATADIR/validPairs/IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz'"
[ ! -f $DATADIR/validPairs/IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz ] && echo -e "Downloading input file from:\n https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/validPairs/IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz" && wget https://files.codeocean.com/files/verified/90f4ed25-715b-4a49-8f70-db31299da83d_v3.0/data/fithic_protocol_data/data/validPairs/IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz && mv IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz $DATDADIR/validPairs

