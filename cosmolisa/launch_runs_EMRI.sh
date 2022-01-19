#!/bin/bash

for n in {GAUSS_1,GAUSS_2,GAUSS_3}; do 
python cosmological_model.py -c EMRI -m CLambdaCDM -d ../data/new_EMRI/EMRI_SAMPLE_MODEL101_TTOT10yr_SIG2_$n -o ~/data1-laghi/cosmolisa/Results/new_EMRI/CLambdaCDM_SNR_100/M101_$n/ --snr_threshold 100.0 --threads 34; done