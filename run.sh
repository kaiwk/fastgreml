#!/usr/bin/env sh

BUILD_TYPE=RelWithDebInfo

./build/${BUILD_TYPE}/fastgreml --grmlist /home/kai/WestlakeProjects/ldms-data/grmlist.txt \
    --mphe 2,/home/kai/WestlakeProjects/ldms-data/mphefile.txt \
    --cov /home/kai/WestlakeProjects/ldms-data/UKB_All_covariates.covar \
    --initial /home/kai/WestlakeProjects/ldms-data/initial_value.txt \
    --output /home/kai/WestlakeProjects/ldms-data/outfile.txt
