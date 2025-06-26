#!/bin/bash

#slurm options
#SBATCH --job-name         fastgreml_huge_mem
#SBATCH --qos              hmem
#SBATCH --partition        intel-fat
#SBATCH --output           fastgreml_huge_mem-%j.out
#SBATCH --threads-per-core 1
#SBATCH --ntasks           1
#SBATCH --cpus-per-task    20
#SBATCH --mem              450G
#SBATCH --mail-type        ALL
#SBATCH --mail-user        wangkai@westlake.edu.cn

module load cmake eigen intelmkl

# BUILD_TYPE=Debug
BUILD_TYPE=Release
# BUILD_TYPE=RelWithDebInfo

export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=20
export OMP_PROC_BIND=close       # keeps threads close together (better L2/L3 sharing, often better for matrix-heavy tasks)
export OMP_PLACES=cores          # threads bound to a core

# export MKL_NUM_THREADS=1
# export MKL_DYNAMIC=false

export KMP_AFFINITY=verbose

LD_PRELOAD="/storage/yangjianLab/wangkai/Repo/mimalloc/out/Release/libmimalloc.so"  \
    MIMALLOC_VERBOSE=1 \
    MIMALLOC_SHOW_STATS=1 \
    MIMALLOC_EAGER_COMMIT=1 \
    MIMALLOC_ALLOW_LARGE_OS_PAGES=1 \
    MIMALLOC_RESERVE_HUGE_OS_PAGES=1 \
    MIMALLOC_PURGE_DECOMMITS=0 \
    ./build/${BUILD_TYPE}/fastgreml \
    --grmlist /storage/yangjianLab/xuting/data/genotype/WES_350k/mgrm_ldms_g8.txt  \
    --mphe 1,/storage/yangjianLab/xuting/data/genotype/300k/nm96_349660.txt \
    --cov /storage/yangjianLab/xuting/ukb/UKB_All_covariates.covar \
    --initial /storage/yangjianLab/xuting/data/genotype/WES_350k/mhe_ldms_g8_nm96_change.txt \
    --output /storage/yangjianLab/wangkai/fastgreml_output.txt
