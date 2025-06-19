echo `date`

# BUILD_TYPE=Debug
BUILD_TYPE=Release
# BUILD_TYPE=RelWithDebInfo

export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close       # keeps threads close together (better L2/L3 sharing, often better for matrix-heavy tasks)
export OMP_PLACES=cores          # threads bound to a core

# export MKL_NUM_THREADS=1
# export MKL_DYNAMIC=false

export KMP_AFFINITY=verbose

LD_PRELOAD="/storage/yangjianLab/wangkai/Repo/mimalloc/out/Release/libmimalloc.so"  \
MIMALLOC_VERBOSE=1 \
# perf record \
#     -g \
#     -e cycles \
#     --sample-cpu \
#     -m 8M \
#     --aio \
#     --call-graph dwarf \
./build/${BUILD_TYPE}/fastgreml --grmlist /storage/yangjianLab/xuting/data/grm/WGS_unrel/sample50k/mgrm_nml_noIG_12group.txt \
--mphe 1,/storage/yangjianLab/xuting/data/phe/50.pheno \
--cov /storage/yangjianLab/xuting/ukb/UKB_All_covariates.covar \
--initial /storage/yangjianLab/wangkai/fastgreml_init_vals.txt \
--output  /storage/yangjianLab/wangkai/fastgreml_output.txt

# perf record --call-graph=fp -F 99 -g  \
# ./build/${BUILD_TYPE}/fastgreml --grmlist /storage/yangjianLab/xuting/data/grm/WGS_unrel/sample50k/mgrm_nml_noIG_12group.txt --mphe 1,/storage/yangjianLab/xuting/data/phe/50.pheno --cov /storage/yangjianLab/xuting/ukb/UKB_All_covariates.covar --initial /storage/yangjianLab/wangkai/fastgreml_init_vals.txt --output  /storage/yangjianLab/wangkai/fastgreml_output.txt

echo `date`
