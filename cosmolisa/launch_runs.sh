#!/bin/bash

declare -a StringArray=(
"-d ../data/EMRIs/EMRI_SAMPLE_MODEL101_TTOT10yr_SIG2_GAUSS -o DEBUG_RUN/ 
--dl_cutoff 2200 --redshift_prior 0 -c EMRI -j 300 --nlive 5000 --poolsize 256 --maxmcmc 1000 --threads 2"
"-d ../data/EMRIs/EMRI_SAMPLE_MODEL101_TTOT10yr_SIG2_GAUSS -o DEBUG_RUN2/ 
--dl_cutoff 2200 --redshift_prior 1 -c EMRI -j 300 --nlive 5000 --poolsize 256 --maxmcmc 1000 --threads 2"
)

# for file in "${StringArray[@]}"; do
#     echo launching $file
#     screen -S $file -d -m python cosmological_model.py $file \
#    -c EMRI \
#    -j 300 \
#    --nlive 5000 --poolsize 256 --maxmcmc 1000 --threads 2 \
#    $file;
# done

STR="-c EMRI -j 300 -d ../data/EMRIs/EMRI_SAMPLE_MODEL101_TTOT10yr_SIG2_GAUSS -o DEBUG_RUN/ \
--dl_cutoff 2200 --redshift_prior 0 --nlive 5000 --poolsize 256 --maxmcmc 1000 --threads 2"

screen -S test -d -m python cosmological_model.py

# for n in {gw151012,gw151226,gw170608}; do 
# screen -S $n -d -m python cbcmodel.py --config-file /home/delpozzo/data2-delpozzo/MLCatalog/gwtc1_$n.ini; done