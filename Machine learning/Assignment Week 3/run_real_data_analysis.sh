
#!/bin/bash
set -e

test_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
neusomatic_dir="$( dirname ${test_dir} )"

cd ${test_dir} #
mkdir -p example
cd example
if [ ! -f GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna ]
then
        if [ ! -f GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz ]
        then
                wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz
        fi
        gunzip -f GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz
fi
if [ ! -f GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.fai ]
then
        samtools faidx GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna
fi
rm -rf work_standalone
#Stand-alone NeuSomatic test
python ${neusomatic_dir}/neusomatic/python/preprocess.py \
        --mode call \
        --reference GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna \
        --region_bed ${neusomatic_dir}/resources/hg38.bed \
        --tumor_bam ${test_dir}/HG003.GRCh38.2x250.bam \
        --normal_bam ${test_dir}/SRR2020635.bam \
        --work work_standalone \
        --scan_maf 0.05 \
        --min_mapq 10 \
        --snp_min_af 0.05 \
        --snp_min_bq 20 \
        --snp_min_ao 10 \
        --ins_min_af 0.05 \
        --del_min_af 0.05 \
        --num_threads 1 \
        --scan_alignments_binary ${neusomatic_dir}/neusomatic/bin/scan_alignments

CUDA_VISIBLE_DEVICES= python ${neusomatic_dir}/neusomatic/python/call.py \
                --candidates_tsv work_standalone/dataset/*/candidates*.tsv \
                --reference GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna \
                --out work_standalone \
                --checkpoint ${neusomatic_dir}/neusomatic/models/NeuSomatic_v0.1.3_standalone_Dream3.pth \
                --num_threads 1 \
                --batch_size 100

python ${neusomatic_dir}/neusomatic/python/postprocess.py \
                --reference GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna \
                --tumor_bam ${test_dir}/HG003.GRCh38.2x250.bam \
                --pred_vcf work_standalone/pred.vcf \
                --candidates_vcf work_standalone/work_tumor/filtered_candidates.vcf \
                --output_vcf work_standalone/NeuSomatic_standalone.vcf \
                --work work_standalone


cd ..
echo "### NeuSomatic stand-alone: SUCCESS! ###"