#!/bin/bash

# TO CHOOSE FROM DISCRETE LIST:
###plist=(62 19 7)
###for S in ${plist[@]};

# TO RUN MBH:

for S in {1..22};
do python cosmological_model.py -c MBH -d ../data/MBHs/popIII/cat_${S}/ -o ../../../data1-laghi/cosmolisa/Results/MBHs/popIII/LambdaCDM/redshift_prior_0/cat_${S}/ --nlive 5000 --poolsize 256 --maxmcmc 1000 --threads 24 --redshift_prior 0;
done

for S in {1..22};
do python cosmological_model.py -c MBH -d ../data/MBHs/popIII/cat_${S}/ -o ../../../data1-laghi/cosmolisa/Results/MBHs/popIII/LambdaCDM/redshift_prior_1/cat_${S}/ --nlive 5000 --poolsize 256 --maxmcmc 1000 --threads 24 --redshift_prior 1;
done

# TO RUN EMRI WITH CATALOG REDUCTION:
# for S in {1..5};
# do python cosmological_model.py -c EMRI -m LambdaCDM -d ./data/EMRIs/EMRI_SAMPLE_MODEL105_TTOT10yr_SIG2_GAUSS/ -o ./data/EMRIs/EMRI_SAMPLE_MODEL105_TTOT10yr_SIG2_GAUSS/LambdaCDM/4yrs/z_04_${S}/ --reduced_catalog=1 --em_selection=1 -j 281$
# done

#do open /Users/danny/Repos/ringdown/ringdown/Area_test_random_${S}_nGR_inj_nGR_rec_zeronoise1_alpha_Bekenstein/Plots/Kerr_intrinsic_corner.png;
#do open /Users/danny/Repos/ringdown/Results/Area_test_zeronoise1/Area_test_zeronoise1_100_events_alpha_Bekenstein/Area_test_random_${S}_nGR_inj_GR_rec_zeronoise1_alpha_Bekenstein/Plots/Kerr_intrinsic_corner.png;
#done

#declare -a arr=("alpha_Bekenstein" "alpha_Davidson" "alpha_Hod" "alpha_Mukhanov")
#events=(1)
#for S in ${events[@]};
#do for T in "${arr[@]}";
#do open /Users/danny/Repos/ringdown/ringdown/SPECIFIC_TESTS/Area_test_random_${S}_nGR_inj_nGR_rec_zeronoise1_${T}_fix_alpha/Plots/Kerr_intrinsic_corner.png;
#done done
