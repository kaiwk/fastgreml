#!/usr/bin/env sh

BUILD_TYPE=RelWithDebInfo
# BUILD_TYPE=Release

LDMS_DATA_PATH=/home/kai/WestlakeProjects/hpc-ldms-data/ldms-data

# see: https://theartofhpc.com/pcse/omp-affinity.html
export OMP_NUM_THREADS=20
export OMP_DISPLAY_ENV=true
export OMP_PROC_BIND=close       # keeps threads close together (better L2/L3 sharing, often better for matrix-heavy tasks)
export OMP_PLACES=cores          # threads bound to a core

# export MKL_NUM_THREADS=1
# export MKL_DYNAMIC=false

export KMP_AFFINITY=verbose

export MIMALLOC_VERBOSE=1
export MIMALLOC_SHOW_STATS=1
export MIMALLOC_EAGER_COMMIT=1
export MIMALLOC_ALLOW_LARGE_OS_PAGES=1
export MIMALLOC_RESERVE_HUGE_OS_PAGES=1
export MIMALLOC_PURGE_DECOMMITS=0  # keep memory hot

sudo -E LD_PRELOAD=/usr/lib/libmimalloc.so \
    perf record \
    -e cycles \
    -e sched:sched_switch --switch-events \
    --sample-cpu \
    -m 8M \
    --aio -z \
    -g --call-graph dwarf \
    ./build/${BUILD_TYPE}/fastgreml --grmlist "${LDMS_DATA_PATH}/mgrm_nml_noIG_12group.txt" \
    --mphe 1,${LDMS_DATA_PATH}/50.pheno \
    --cov ${LDMS_DATA_PATH}/UKB_All_covariates.covar \
    --initial ${LDMS_DATA_PATH}/fastgreml_init_vals.txt \
    --output ${LDMS_DATA_PATH}/outfile.txt
