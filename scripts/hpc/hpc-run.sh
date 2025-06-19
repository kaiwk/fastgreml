echo `date`

# BUILD_TYPE=Debug
# BUILD_TYPE=Release
BUILD_TYPE=RelWithDebInfo

perf record --call-graph=dwarf -F 99 -g  \
./build/${BUILD_TYPE}/fastgreml --grmlist /storage/yangjianLab/xuting/data/grm/WGS_unrel/sample50k/mgrm_nml_noIG_12group.txt \
--mphe 1,/storage/yangjianLab/xuting/data/phe/50.pheno \
--cov /storage/yangjianLab/xuting/ukb/UKB_All_covariates.covar \
--initial /storage/yangjianLab/wangkai/fastgreml_init_vals.txt \
--output  /storage/yangjianLab/wangkai/fastgreml_output.txt

# perf record --call-graph=fp -F 99 -g  \
# ./build/${BUILD_TYPE}/fastgreml --grmlist /storage/yangjianLab/xuting/data/grm/WGS_unrel/sample50k/mgrm_nml_noIG_12group.txt --mphe 1,/storage/yangjianLab/xuting/data/phe/50.pheno --cov /storage/yangjianLab/xuting/ukb/UKB_All_covariates.covar --initial /storage/yangjianLab/wangkai/fastgreml_init_vals.txt --output  /storage/yangjianLab/wangkai/fastgreml_output.txt

echo `date`
